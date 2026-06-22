"""Topology expansion utilities for Neuro-FSAI candidate sparsity patterns."""

import numpy as np
import scipy.sparse as sp
import torch


def _as_unit_csr_from_edge_index(edge_index, num_nodes):
    row = edge_index[0].detach().cpu().numpy()
    col = edge_index[1].detach().cpu().numpy()
    values = np.ones(row.shape[0], dtype=bool)
    return sp.csr_matrix((values, (row, col)), shape=(num_nodes, num_nodes))


def expand_topology_to_2hop_scipy(edge_index, edge_attr, num_nodes,
                                  max_topology_edges=0,
                                  max_topology_ratio=0.0):
    """Expand directed topology to union of 1-hop and 2-hop candidate edges.

    New 2-hop-only edges receive zero edge attributes, so they are candidate
    locations for learned FSAI factors but do not change the represented A.
    When the expanded topology exceeds the provided caps, the original topology
    is returned to avoid memory blowups.
    """
    device = edge_index.device
    original_edges = int(edge_index.shape[1])
    A_sp = _as_unit_csr_from_edge_index(edge_index, num_nodes)

    A_sq_sp = A_sp @ A_sp
    A_sq_sp.setdiag(False)
    A_sq_sp.eliminate_zeros()

    A_new_sp = A_sq_sp > A_sp
    new_row, new_col = A_new_sp.nonzero()
    if new_row.size == 0:
        return edge_index, edge_attr, False

    expanded_edges = original_edges + int(new_row.size)
    if max_topology_edges and max_topology_edges > 0:
        if expanded_edges > max_topology_edges:
            return edge_index, edge_attr, False
    if max_topology_ratio and max_topology_ratio > 0:
        if expanded_edges > int(max_topology_ratio * max(1, original_edges)):
            return edge_index, edge_attr, False

    new_edge_index = torch.as_tensor(
        np.vstack([new_row, new_col]),
        dtype=edge_index.dtype,
        device=device)
    new_edge_attr = torch.zeros(
        (new_edge_index.shape[1], edge_attr.shape[1]),
        dtype=edge_attr.dtype,
        device=device)

    combined_edge_index = torch.cat([edge_index, new_edge_index], dim=1)
    combined_edge_attr = torch.cat([edge_attr, new_edge_attr], dim=0)
    return combined_edge_index, combined_edge_attr, True


def expand_sparse_topology_scipy(A, topology_hop=1,
                                 max_topology_edges=0,
                                 max_topology_ratio=0.0):
    """Return a scipy CSR candidate topology for A or union(A, A^2)."""
    pattern = A.copy().tocsr()
    pattern.data = np.ones_like(pattern.data, dtype=bool)
    pattern.eliminate_zeros()
    if topology_hop == 1:
        return pattern, False
    if topology_hop != 2:
        raise ValueError(f"Unsupported topology_hop={topology_hop}; expected 1 or 2")

    squared = pattern @ pattern
    squared.setdiag(False)
    squared.eliminate_zeros()

    expanded = pattern + squared
    expanded.data = np.ones_like(expanded.data, dtype=bool)
    expanded.eliminate_zeros()
    expanded = expanded.tocsr()

    if max_topology_edges and max_topology_edges > 0 and expanded.nnz > max_topology_edges:
        return pattern, False
    if max_topology_ratio and max_topology_ratio > 0:
        if expanded.nnz > int(max_topology_ratio * max(1, pattern.nnz)):
            return pattern, False
    return expanded, expanded.nnz > pattern.nnz
