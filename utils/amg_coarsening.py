"""Deterministic AMG-style graph coarsening utilities.

This module intentionally keeps coarsening outside the learned model.  The
projection matrices are a numerical setup artifact, like RCM/equilibration,
and depend only on the input sparse matrix A.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch


@dataclass(frozen=True)
class AMGLevel:
    """One fine-to-coarse transition in an AMG hierarchy."""

    A_fine: sp.csr_matrix
    P: sp.csr_matrix
    A_coarse: sp.csr_matrix


def _canonical_csr(A: sp.spmatrix) -> sp.csr_matrix:
    A = A.tocsr()
    A.sum_duplicates()
    A.eliminate_zeros()
    return A


def build_amg_projection_hem(A: sp.spmatrix) -> sp.csr_matrix:
    """Build an unsmoothed aggregation projection with greedy HEM.

    Args:
        A: Fine-level sparse matrix in CSR-compatible format.

    Returns:
        P with shape ``[n_fine, n_coarse]``.  Each fine node maps to exactly
        one coarse aggregate.  Columns are coarse super-nodes.
    """
    A = _canonical_csr(A)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"AMG coarsening requires a square matrix, got {A.shape}")

    A_abs = A.copy()
    A_abs.data = np.abs(A_abs.data)
    strength = (A_abs + A_abs.T).tocsr()
    strength.setdiag(0.0)
    strength.eliminate_zeros()

    matched = np.zeros(n, dtype=bool)
    node_to_coarse = np.full(n, -1, dtype=np.int64)
    p_rows: list[int] = []
    p_cols: list[int] = []
    coarse_idx = 0

    # Edge-centric greedy matching is more aggressive than row-order matching:
    # sort heavy couplings globally, then accept an edge iff both endpoints are
    # still unmatched.  Keep only one orientation of the symmetric graph.
    strength_coo = strength.tocoo()
    upper = strength_coo.row < strength_coo.col
    edge_row = strength_coo.row[upper].astype(np.int64, copy=False)
    edge_col = strength_coo.col[upper].astype(np.int64, copy=False)
    edge_weight = strength_coo.data[upper]
    if edge_weight.size:
        order = np.argsort(-edge_weight, kind="mergesort")
        for edge_idx in order:
            i = int(edge_row[edge_idx])
            j = int(edge_col[edge_idx])
            if matched[i] or matched[j]:
                continue
            p_rows.extend([i, j])
            p_cols.extend([coarse_idx, coarse_idx])
            matched[i] = True
            matched[j] = True
            node_to_coarse[i] = coarse_idx
            node_to_coarse[j] = coarse_idx
            coarse_idx += 1

    indptr = strength.indptr
    indices = strength.indices
    data = strength.data

    # Attach leftover nodes to the strongest already-created neighboring
    # aggregate.  This is more AMG-like than forcing every unmatched node to be
    # a singleton, and gives the hierarchy enough depth on sparse circuit graphs.
    for i in np.flatnonzero(~matched):
        start, end = indptr[i], indptr[i + 1]
        neighbors = indices[start:end]
        weights = data[start:end]
        neighbor_coarse = node_to_coarse[neighbors]
        valid = neighbor_coarse >= 0
        if np.any(valid):
            target_col = int(neighbor_coarse[valid][int(np.argmax(weights[valid]))])
        else:
            target_col = coarse_idx
            coarse_idx += 1
        p_rows.append(int(i))
        p_cols.append(target_col)
        matched[i] = True
        node_to_coarse[i] = target_col

    p_data = np.ones(len(p_rows), dtype=np.float32)
    P = sp.csr_matrix((p_data, (p_rows, p_cols)), shape=(n, coarse_idx))
    P.sum_duplicates()
    P.eliminate_zeros()
    return P


def galerkin_coarse_matrix(A: sp.spmatrix, P: sp.spmatrix) -> sp.csr_matrix:
    """Compute the Galerkin coarse operator ``A_c = P.T @ A @ P``."""
    A = _canonical_csr(A)
    P = _canonical_csr(P)
    A_coarse = (P.T @ A @ P).tocsr()
    A_coarse.sum_duplicates()
    A_coarse.eliminate_zeros()
    return A_coarse


def build_amg_hierarchy(
    A: sp.spmatrix,
    *,
    max_levels: int = 3,
    min_coarse_nodes: int = 500,
) -> list[AMGLevel]:
    """Build a deterministic AMG hierarchy with greedy HEM.

    The returned list contains transitions from fine to coarse.  The final
    coarse matrix is the ``A_coarse`` of the last level.
    """
    if max_levels < 1:
        return []

    levels: list[AMGLevel] = []
    current = _canonical_csr(A)
    for _ in range(max_levels):
        if current.shape[0] <= min_coarse_nodes:
            break
        P = build_amg_projection_hem(current)
        if P.shape[1] >= current.shape[0]:
            break
        coarse = galerkin_coarse_matrix(current, P)
        levels.append(AMGLevel(A_fine=current, P=P, A_coarse=coarse))
        current = coarse
    return levels


def scipy_csr_to_torch_sparse(
    matrix: sp.spmatrix,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert a SciPy sparse matrix to a coalesced torch COO sparse tensor."""
    coo = _canonical_csr(matrix).tocoo()
    indices = torch.from_numpy(
        np.vstack([coo.row.astype(np.int64), coo.col.astype(np.int64)])
    )
    values = torch.from_numpy(coo.data).to(dtype=dtype)
    out = torch.sparse_coo_tensor(indices, values, coo.shape, dtype=dtype)
    out = out.coalesce()
    if device is not None:
        out = out.to(device)
    return out


def projection_column_counts(P: sp.spmatrix) -> np.ndarray:
    """Return aggregate sizes for a binary projection matrix P."""
    P = _canonical_csr(P)
    return np.asarray(P.sum(axis=0)).ravel().astype(np.float32, copy=False)


def restrict_features(
    P_t: torch.Tensor,
    x_fine: torch.Tensor,
    counts: torch.Tensor | None = None,
    *,
    average: bool = True,
) -> torch.Tensor:
    """Restrict fine node features to coarse node features with ``P.T @ x``."""
    x_coarse = torch.sparse.mm(P_t.transpose(0, 1), x_fine)
    if average:
        if counts is None:
            counts = torch.sparse.sum(P_t, dim=0).to_dense()
        x_coarse = x_coarse / counts.to(device=x_coarse.device, dtype=x_coarse.dtype).clamp_min(1.0).unsqueeze(-1)
    return x_coarse


def prolong_features(P_t: torch.Tensor, x_coarse: torch.Tensor) -> torch.Tensor:
    """Prolong coarse node features back to the fine grid with ``P @ x``."""
    return torch.sparse.mm(P_t, x_coarse)
