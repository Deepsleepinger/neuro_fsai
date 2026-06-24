import argparse
import json
import pathlib
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import LinearOperator, bicgstab, eigsh, spilu


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model_neuro_fsai import PDEDirectedConv
from utils.convert_suitesparse import (
    algebraic_graph_features,
    canonicalize_sparse_matrix,
    read_mtx_from_tar,
    values_on_pattern,
)
from utils.amg_coarsening import (
    build_amg_hierarchy,
    projection_column_counts,
    prolong_features,
    restrict_features,
    scipy_csr_to_torch_sparse,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-matrix Neuro-SPAI overfit against SPILU inverse top-k labels.")
    parser.add_argument("--matrix-tar", required=True)
    parser.add_argument(
        "--prepared-data-dir",
        default=None,
        help="Optional directory containing train/train_0000.npy; only its RHS is used.")
    parser.add_argument("--save-dir", default="results/local_checkpoints")
    parser.add_argument("--exp-name", default="single_neuro_spai")
    parser.add_argument("--load-checkpoint", default=None)
    parser.add_argument("--no-teacher", action="store_true")
    parser.add_argument("--row-topk", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-iterations", type=int, default=2)
    parser.add_argument("--decoder-type", choices=["mlp", "bilinear"], default="mlp")
    parser.add_argument("--use-node-embedding", action="store_true")
    parser.add_argument("--weight-abs", type=float, default=5.0)
    parser.add_argument("--eval-damping-grid", default="1.0,0.5,0.25,0.1,0.05,0.01")
    parser.add_argument("--feature-mode", choices=["algebraic", "legacy"], default="algebraic")
    parser.add_argument("--target-transform", choices=["linear", "signed_log10"], default="linear")
    parser.add_argument("--target-scale-mode", choices=["teacher", "jacobi", "unit"], default="teacher")
    parser.add_argument("--base-mode", choices=["jacobi", "identity", "zero"], default="jacobi")
    parser.add_argument("--log-output-clip", type=float, default=16.0)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--hutchinson-weight", type=float, default=0.0)
    parser.add_argument("--hutchinson-probes", type=int, default=4)
    parser.add_argument("--equilibrate", action="store_true")
    parser.add_argument("--equil-iters", type=int, default=5)
    parser.add_argument("--equil-eps", type=float, default=1e-12)
    parser.add_argument("--reorder", choices=["none", "rcm"], default="none")
    parser.add_argument("--spectral-pe-dim", type=int, default=0)
    parser.add_argument("--topology-hop", type=int, choices=[1, 2], default=1)
    parser.add_argument("--topology-drop-tol", type=float, default=0.0)
    parser.add_argument("--topology-row-topk", type=int, default=64)
    parser.add_argument("--amg-levels", type=int, default=0)
    parser.add_argument("--amg-min-coarse-nodes", type=int, default=500)
    parser.add_argument("--val-freq", type=int, default=25)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rhs-seed", type=int, default=20260622)
    parser.add_argument("--spilu-drop-tol", type=float, default=1e-4)
    parser.add_argument("--spilu-fill-factor", type=float, default=10.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--rtol", type=float, default=1e-8)
    return parser.parse_args()


def make_mlp(in_dim, out_dim, hidden_dim, hidden_layers, norm=False):
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(max(0, hidden_layers - 1)):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    if norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


class NeuroSPAI(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_iterations,
                 num_nodes=None, use_node_embedding=False, decoder_type="mlp",
                 amg_levels=0):
        super().__init__()
        self.use_node_embedding = use_node_embedding
        self.decoder_type = decoder_type
        self.amg_levels = int(amg_levels)
        self.node_encoder = make_mlp(node_dim, hidden_dim, hidden_dim, 1, norm=True)
        if use_node_embedding:
            if num_nodes is None:
                raise ValueError("num_nodes is required when use_node_embedding=True")
            self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        else:
            self.node_embedding = None
        self.edge_encoder = make_mlp(edge_dim, hidden_dim, hidden_dim, 1, norm=True)
        self.mp_layers = nn.ModuleList([
            PDEDirectedConv(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_iterations)
        ])
        self.amg_mp_layers = nn.ModuleList([
            PDEDirectedConv(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(max(0, self.amg_levels))
        ])
        self.amg_fuse = (
            make_mlp(2 * hidden_dim, hidden_dim, hidden_dim, 1, norm=True)
            if self.amg_levels > 0 else None
        )
        if decoder_type == "mlp":
            self.edge_decoder = make_mlp(3 * hidden_dim, 1, hidden_dim, 2, norm=False)
            self._zero_init_last()
        elif decoder_type == "bilinear":
            self.W_dir = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.W_edge = nn.Linear(hidden_dim, 1, bias=False)
            self.bias = nn.Parameter(torch.zeros(1))
            nn.init.zeros_(self.W_dir.weight)
            nn.init.zeros_(self.W_edge.weight)
        else:
            raise ValueError(f"unknown decoder_type={decoder_type!r}")

    def _zero_init_last(self):
        for layer in reversed(self.edge_decoder):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                return

    def forward(self, node_attr, edge_index, edge_attr, amg_data=None):
        node_feat = self.node_encoder(node_attr)
        if self.node_embedding is not None:
            node_ids = torch.arange(node_attr.shape[0], device=node_attr.device)
            node_feat = node_feat + self.node_embedding(node_ids)
        edge_feat = self.edge_encoder(edge_attr)
        x = node_feat
        for mp in self.mp_layers:
            x = torch.relu(mp(x, edge_index, edge_feat))
        if self.amg_fuse is not None and amg_data:
            coarse_x = x
            used_levels = []
            for level, mp in zip(amg_data, self.amg_mp_layers):
                coarse_x = restrict_features(
                    level["P"], coarse_x, level.get("counts"), average=True)
                coarse_edge_feat = self.edge_encoder(level["edge_attr"])
                coarse_x = torch.relu(
                    mp(coarse_x, level["edge_index"], coarse_edge_feat))
                used_levels.append(level)
            context = coarse_x
            for level in reversed(used_levels):
                context = prolong_features(level["P"], context)
            x = self.amg_fuse(torch.cat([x, context], dim=-1))
        source, target = edge_index
        if self.decoder_type == "mlp":
            decoded = self.edge_decoder(torch.cat([x[source], x[target], edge_feat], dim=-1))
            return decoded.squeeze(-1)
        src_proj = self.W_dir(x[source])
        node_interaction = torch.sum(src_proj * x[target], dim=-1)
        edge_contribution = self.W_edge(edge_feat).squeeze(-1)
        return node_interaction + edge_contribution + self.bias


def load_rhs(prepared_data_dir, scale, A_scaled, rhs_seed):
    if prepared_data_dir is None:
        rng = np.random.default_rng(rhs_seed)
        x_true = rng.uniform(-1.0, 1.0, A_scaled.shape[0])
        return np.asarray(A_scaled @ x_true, dtype=np.float64).reshape(-1)
    path = pathlib.Path(prepared_data_dir) / "train" / "train_0000.npy"
    data = np.load(path, allow_pickle=True)
    if len(data) != 1:
        raise ValueError(f"Expected one graph in {path}, got {len(data)}")
    return np.asarray(data[0]["rhs"], dtype=np.float64).reshape(-1) / scale


def signed_log10_np(values):
    return np.sign(values) * np.log10(1.0 + np.abs(values))


def inverse_signed_log10_torch(values, clip):
    clipped = torch.clamp(values, min=-clip, max=clip)
    return torch.sign(clipped) * (torch.pow(10.0, torch.abs(clipped)) - 1.0)


def build_spilu_inverse_row_topk(A_scaled, row_topk, drop_tol, fill_factor):
    teacher = spilu(A_scaled.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)
    dense_inverse = np.asarray(teacher.solve(np.eye(A_scaled.shape[0])), dtype=np.float64)
    n = dense_inverse.shape[0]
    topk = min(row_topk, n)
    rows = []
    cols = []
    vals = []
    for row in range(n):
        row_abs = np.abs(dense_inverse[row])
        idx = np.argpartition(row_abs, -topk)[-topk:]
        rows.append(np.full(topk, row, dtype=np.int64))
        cols.append(idx.astype(np.int64, copy=False))
        vals.append(dense_inverse[row, idx])

    target = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()
    target.sum_duplicates()
    target = target.tolil()
    target.setdiag(np.diag(dense_inverse))
    target = target.tocsr()
    target.eliminate_zeros()
    return target, teacher, dense_inverse


def max_abs_equilibrate(A_csr, num_iters, eps):
    """Iterative max-abs diagonal equilibration: A_hat = diag(dr) A diag(dc)."""
    n = A_csr.shape[0]
    A_eq = A_csr.tocsr(copy=True)
    dr = np.ones(n, dtype=np.float64)
    dc = np.ones(n, dtype=np.float64)

    def row_col_max_abs(matrix):
        coo = matrix.tocoo()
        abs_data = np.abs(coo.data)
        row_max = np.zeros(n, dtype=np.float64)
        col_max = np.zeros(n, dtype=np.float64)
        np.maximum.at(row_max, coo.row, abs_data)
        np.maximum.at(col_max, coo.col, abs_data)
        return row_max, col_max

    for _ in range(max(0, num_iters)):
        row_max, _ = row_col_max_abs(A_eq)
        row_scale = 1.0 / np.sqrt(np.maximum(row_max, eps))
        row_scale[row_max <= 0.0] = 1.0
        A_eq = sp.diags(row_scale, format="csr") @ A_eq
        dr *= row_scale

        _, col_max = row_col_max_abs(A_eq)
        col_scale = 1.0 / np.sqrt(np.maximum(col_max, eps))
        col_scale[col_max <= 0.0] = 1.0
        A_eq = A_eq @ sp.diags(col_scale, format="csr")
        dc *= col_scale

    row_max, col_max = row_col_max_abs(A_eq)
    stats = {
        "row_max_min": float(row_max[row_max > 0.0].min()) if np.any(row_max > 0.0) else 0.0,
        "row_max_max": float(row_max.max()) if row_max.size else 0.0,
        "col_max_min": float(col_max[col_max > 0.0].min()) if np.any(col_max > 0.0) else 0.0,
        "col_max_max": float(col_max.max()) if col_max.size else 0.0,
        "dr_min": float(dr.min()) if dr.size else 0.0,
        "dr_max": float(dr.max()) if dr.size else 0.0,
        "dc_min": float(dc.min()) if dc.size else 0.0,
        "dc_max": float(dc.max()) if dc.size else 0.0,
    }
    return A_eq.tocsr(), dr, dc, stats


def matrix_bandwidth(A_csr):
    coo = A_csr.tocoo()
    if coo.nnz == 0:
        return 0
    return int(np.max(np.abs(coo.row.astype(np.int64) - coo.col.astype(np.int64))))


def apply_reordering(A_csr, b, mode):
    """Apply a symmetric graph permutation before graph construction/solving."""
    if mode == "none":
        return A_csr.tocsr(), b, np.arange(A_csr.shape[0], dtype=np.int64), None
    if mode != "rcm":
        raise ValueError(f"unsupported reorder mode={mode!r}")

    pattern = A_csr.copy().tocsr()
    pattern.data = np.ones_like(pattern.data, dtype=np.float64)
    pattern = (pattern + pattern.T).tocsr()
    pattern.data = np.ones_like(pattern.data, dtype=np.float64)
    pattern.setdiag(1.0)
    pattern.eliminate_zeros()

    perm = reverse_cuthill_mckee(pattern, symmetric_mode=True).astype(np.int64, copy=False)
    A_reordered = A_csr[perm, :][:, perm].tocsr()
    b_reordered = None if b is None else np.asarray(b, dtype=np.float64)[perm]
    stats = {
        "mode": mode,
        "bandwidth_before": matrix_bandwidth(A_csr),
        "bandwidth_after": matrix_bandwidth(A_reordered),
    }
    return A_reordered, b_reordered, perm, stats


def row_topk_sparse(A_csr, row_topk):
    if row_topk <= 0:
        return A_csr
    A_csr = A_csr.tocsr()
    rows = []
    cols = []
    data = []
    for row in range(A_csr.shape[0]):
        start, end = A_csr.indptr[row], A_csr.indptr[row + 1]
        length = end - start
        if length == 0:
            continue
        if length <= row_topk:
            keep = np.arange(start, end)
        else:
            local = np.argpartition(np.abs(A_csr.data[start:end]), -row_topk)[-row_topk:]
            keep = start + local
        rows.append(np.full(keep.shape[0], row, dtype=np.int64))
        cols.append(A_csr.indices[keep].astype(np.int64, copy=False))
        data.append(A_csr.data[keep].astype(np.float64, copy=False))
    if not rows:
        return sp.csr_matrix(A_csr.shape, dtype=np.float64)
    return sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=A_csr.shape,
    ).tocsr()


def build_topology_pattern(A_scaled, topology_hop, topology_drop_tol=0.0,
                           topology_row_topk=64):
    A_bool = A_scaled.copy().tocsr()
    A_bool.data = np.ones_like(A_bool.data, dtype=np.float64)
    if topology_hop == 1:
        pattern = A_bool
    elif topology_hop == 2:
        A_sq = (A_scaled @ A_scaled).tocsr()
        A_sq = row_topk_sparse(A_sq, topology_row_topk)
        if topology_drop_tol > 0.0:
            keep = np.abs(A_sq.data) >= topology_drop_tol
            A_sq.data = A_sq.data * keep
            A_sq.eliminate_zeros()
        pattern = (A_bool + A_sq).tocsr()
        pattern.data = np.ones_like(pattern.data, dtype=np.float64)
    else:
        raise ValueError(f"unsupported topology_hop={topology_hop}")
    pattern = pattern.tolil()
    pattern.setdiag(1.0)
    pattern = pattern.tocsr()
    pattern.sum_duplicates()
    pattern.eliminate_zeros()
    return pattern


def spectral_positional_encoding(A_scaled, dim):
    if dim <= 0:
        return np.zeros((A_scaled.shape[0], 0), dtype=np.float32)
    n = A_scaled.shape[0]
    if n <= 2:
        return np.zeros((n, dim), dtype=np.float32)

    pattern = A_scaled.copy().tocsr()
    pattern.data = np.ones_like(pattern.data, dtype=np.float64)
    pattern = (pattern + pattern.T).tocsr()
    pattern.data = np.ones_like(pattern.data, dtype=np.float64)
    pattern.setdiag(0.0)
    pattern.eliminate_zeros()

    eig_count = min(dim + 1, n - 1)
    lap = csgraph_laplacian(pattern, normed=True).astype(np.float64).tocsr()
    _, vecs = eigsh(lap, k=eig_count, which="SM", tol=1e-3, maxiter=max(1000, 20 * n))
    pe = vecs[:, 1:eig_count]
    if pe.shape[1] < dim:
        pe = np.pad(pe, ((0, 0), (0, dim - pe.shape[1])))
    pe = pe[:, :dim]

    for j in range(pe.shape[1]):
        idx = int(np.argmax(np.abs(pe[:, j])))
        if pe[idx, j] < 0.0:
            pe[:, j] *= -1.0
    pe = np.clip(pe * np.sqrt(float(n)), -5.0, 5.0)
    return pe.astype(np.float32, copy=False)


def build_graph_tensors(A_scaled, target_csr, device, feature_mode="algebraic",
                        target_transform="linear", topology_hop=1,
                        topology_drop_tol=0.0, topology_row_topk=64,
                        spectral_pe_dim=0, target_scale_mode="teacher",
                        base_mode="jacobi", amg_levels=0,
                        amg_min_coarse_nodes=500):
    n = A_scaled.shape[0]
    pattern_csr = build_topology_pattern(
        A_scaled, topology_hop, topology_drop_tol, topology_row_topk)
    pattern_coo = pattern_csr.tocoo()
    row = pattern_coo.row.astype(np.int64, copy=False)
    col = pattern_coo.col.astype(np.int64, copy=False)
    target_values = values_on_pattern(target_csr, row, col)
    A_values = values_on_pattern(A_scaled, row, col)
    diag = A_scaled.diagonal().astype(np.float64)
    safe_diag = diag.copy()
    safe_diag[safe_diag == 0.0] = 1.0
    base_values = np.zeros_like(target_values)
    diag_mask = row == col
    if base_mode == "jacobi":
        base_values[diag_mask] = 1.0 / safe_diag[row[diag_mask]]
    elif base_mode == "identity":
        base_values[diag_mask] = 1.0
    elif base_mode == "zero":
        pass
    else:
        raise ValueError(f"unknown base_mode={base_mode!r}")
    residual_values = target_values - base_values
    if target_transform == "linear":
        if target_scale_mode == "teacher":
            scale_values = residual_values
        elif target_scale_mode == "jacobi":
            scale_values = base_values
        elif target_scale_mode == "unit":
            scale_values = np.ones(1, dtype=np.float64)
        else:
            raise ValueError(f"unknown target_scale_mode={target_scale_mode!r}")
        target_scale = max(1e-12, float(np.max(np.abs(scale_values))))
        target_norm = residual_values / target_scale
    elif target_transform == "signed_log10":
        target_scale = 1.0
        target_norm = signed_log10_np(residual_values)
    else:
        raise ValueError(f"unknown target_transform={target_transform!r}")

    if feature_mode == "algebraic":
        node_attr, edge_attr = algebraic_graph_features(A_scaled, row, col, A_values)
    elif feature_mode == "legacy":
        row_degree = np.diff(pattern_csr.indptr).astype(np.float32)
        max_degree = max(1.0, float(row_degree.max()))
        node_attr = np.zeros((n, 4), dtype=np.float32)
        node_attr[:, 0] = np.arange(n, dtype=np.float32) / max(1, n - 1) * 2.0 - 1.0
        node_attr[:, 1] = row_degree / max_degree * 2.0 - 1.0
        node_attr[:, 2] = diag.astype(np.float32)
        node_attr[:, 3] = 0.0

        edge_attr = np.zeros((row.shape[0], 3), dtype=np.float32)
        edge_attr[:, 0] = (np.abs(row - col).astype(np.float32) / n) * 2.0 - 1.0
        edge_attr[:, 1] = (row == col).astype(np.float32)
        edge_attr[:, 2] = A_values.astype(np.float32)
    else:
        raise ValueError(f"unknown feature_mode={feature_mode!r}")
    if spectral_pe_dim > 0:
        node_attr = np.concatenate(
            [node_attr, spectral_positional_encoding(A_scaled, spectral_pe_dim)],
            axis=1,
        )
    if topology_hop == 2:
        A_sq_values = values_on_pattern((A_scaled @ A_scaled).tocsr(), row, col)
        abs_diag = np.abs(diag)
        sq_scale = np.sqrt(abs_diag[row] * abs_diag[col] + 1e-12)
        A_sq_rel = np.clip(A_sq_values / sq_scale, -10.0, 10.0).astype(np.float32)
    else:
        A_sq_rel = np.zeros(row.shape[0], dtype=np.float32)
    edge_attr = np.concatenate([edge_attr, A_sq_rel[:, None]], axis=1)
    edge_index = np.stack([col, row], axis=0).astype(np.int64)
    A_coo = A_scaled.tocoo()
    A_indices = np.stack([
        A_coo.row.astype(np.int64, copy=False),
        A_coo.col.astype(np.int64, copy=False),
    ], axis=0)
    amg_data = []
    if amg_levels > 0:
        hierarchy = build_amg_hierarchy(
            A_scaled,
            max_levels=amg_levels,
            min_coarse_nodes=amg_min_coarse_nodes,
        )
        edge_dim = edge_attr.shape[1]
        for level in hierarchy:
            A_c = level.A_coarse.tocoo()
            c_row = A_c.row.astype(np.int64, copy=False)
            c_col = A_c.col.astype(np.int64, copy=False)
            c_values = A_c.data.astype(np.float64, copy=False)
            _, c_edge_attr = algebraic_graph_features(
                level.A_coarse, c_row, c_col, c_values)
            if c_edge_attr.shape[1] < edge_dim:
                pad = np.zeros(
                    (c_edge_attr.shape[0], edge_dim - c_edge_attr.shape[1]),
                    dtype=np.float32,
                )
                c_edge_attr = np.concatenate([c_edge_attr, pad], axis=1)
            elif c_edge_attr.shape[1] > edge_dim:
                c_edge_attr = c_edge_attr[:, :edge_dim]
            c_edge_index = np.stack([c_col, c_row], axis=0).astype(np.int64)
            amg_data.append({
                "P": scipy_csr_to_torch_sparse(level.P, device=device),
                "counts": torch.from_numpy(
                    projection_column_counts(level.P)).to(device),
                "edge_index": torch.from_numpy(c_edge_index).to(device),
                "edge_attr": torch.from_numpy(c_edge_attr).to(device),
                "fine_nodes": int(level.P.shape[0]),
                "coarse_nodes": int(level.P.shape[1]),
                "coarse_nnz": int(level.A_coarse.nnz),
            })

    return {
        "node_attr": torch.from_numpy(node_attr).to(device),
        "edge_attr": torch.from_numpy(edge_attr).to(device),
        "edge_index": torch.from_numpy(edge_index).to(device),
        "target_norm": torch.from_numpy(target_norm.astype(np.float32)).to(device),
        "base_values": torch.from_numpy(base_values.astype(np.float32)).to(device),
        "base_values64": base_values.astype(np.float64, copy=False),
        "target_transform": target_transform,
        "target_scale_mode": target_scale_mode,
        "target_scale": float(target_scale),
        "base_mode": base_mode,
        "row": row,
        "col": col,
        "target_values": target_values,
        "base_values_np": base_values,
        "safe_diag": safe_diag,
        "topology_hop": int(topology_hop),
        "topology_drop_tol": float(topology_drop_tol),
        "topology_row_topk": int(topology_row_topk),
        "spectral_pe_dim": int(spectral_pe_dim),
        "amg_data": amg_data,
        "amg_level_shapes": [
            {
                "fine_nodes": level["fine_nodes"],
                "coarse_nodes": level["coarse_nodes"],
                "coarse_nnz": level["coarse_nnz"],
            }
            for level in amg_data
        ],
        "support_nnz": int(pattern_csr.nnz),
        "G_sparse_index": torch.from_numpy(np.stack([row, col], axis=0)).long().to(device),
        "A_sparse_index": torch.from_numpy(A_indices).long().to(device),
        "A_sparse_values": torch.from_numpy(A_coo.data.astype(np.float32)).to(device),
    }


def csr_from_values(row, col, values, shape):
    G = sp.coo_matrix((values.astype(np.float64, copy=False), (row, col)), shape=shape).tocsr()
    G.sum_duplicates()
    G.eliminate_zeros()
    return G


def run_bicgstab(A_csr, b, apply_prec, max_iter, rtol):
    counter = {"iters": 0}
    prec_stats = {"calls": 0, "time": 0.0}
    norm0 = max(1e-30, float(np.linalg.norm(b)))

    def callback(_xk):
        counter["iters"] += 1

    def timed_prec(v):
        start = time.perf_counter()
        out = apply_prec(v)
        prec_stats["time"] += time.perf_counter() - start
        prec_stats["calls"] += 1
        return out

    op = LinearOperator(A_csr.shape, matvec=timed_prec, dtype=np.float64)
    start = time.perf_counter()
    x, info = bicgstab(
        A_csr,
        b,
        x0=np.zeros(A_csr.shape[0], dtype=np.float64),
        rtol=rtol,
        atol=0.0,
        maxiter=max_iter,
        M=op,
        callback=callback,
    )
    solve_time = time.perf_counter() - start
    final_rel = float(np.linalg.norm(b - A_csr @ x) / norm0)
    return {
        "iterations": int(counter["iters"] if info == 0 else max_iter),
        "raw_callback_iterations": int(counter["iters"]),
        "solve_time": float(solve_time),
        "preconditioner_time": float(prec_stats["time"]),
        "preconditioner_calls": int(prec_stats["calls"]),
        "preconditioner_avg_time": float(
            prec_stats["time"] / max(1, prec_stats["calls"])),
        "info": int(info),
        "final_rel": final_rel,
    }


def evaluate_values(A_scaled, b, values, graph, max_iter, rtol):
    G = csr_from_values(graph["row"], graph["col"], values, A_scaled.shape)
    return run_bicgstab(A_scaled, b, lambda v: G @ v, max_iter, rtol), G


def decode_residual(pred_norm, graph, log_output_clip):
    if graph["target_transform"] == "linear":
        return pred_norm * graph["target_scale"]
    return inverse_signed_log10_torch(pred_norm, log_output_clip)


def hutchinson_loss(pred_norm, graph, log_output_clip, probes, generator, probe_vectors=None):
    if probes <= 0 and probe_vectors is None:
        return pred_norm.new_zeros(())
    n = graph["node_attr"].shape[0]
    residual = decode_residual(pred_norm, graph, log_output_clip)
    values = graph["base_values"] + residual
    G = torch.sparse_coo_tensor(
        graph["G_sparse_index"],
        values,
        (n, n),
        device=values.device,
        dtype=values.dtype,
    ).coalesce()
    A = torch.sparse_coo_tensor(
        graph["A_sparse_index"],
        graph["A_sparse_values"].to(dtype=values.dtype),
        (n, n),
        device=values.device,
        dtype=values.dtype,
    ).coalesce()
    if probe_vectors is None:
        v = torch.randint(
            0, 2, (n, probes),
            device=values.device,
            generator=generator,
            dtype=torch.int64,
        )
        v = (v * 2 - 1).to(dtype=values.dtype)
    else:
        v = probe_vectors.to(device=values.device, dtype=values.dtype)
        if v.dim() == 1:
            v = v[:, None]
        if v.shape[0] != n:
            raise ValueError(f"probe_vectors has {v.shape[0]} rows, expected {n}")
    Av = torch.sparse.mm(A, v)
    GAv = torch.sparse.mm(G, Av)
    denom = v.pow(2).mean().clamp_min(1e-12)
    return (GAv - v).pow(2).mean() / denom


def metric_key(result):
    failed = 1 if result["info"] != 0 else 0
    final_rel = result["final_rel"]
    if not np.isfinite(final_rel):
        final_rel = float("inf")
    return (failed, result["iterations"], final_rel)


def parse_float_list(text):
    return [float(x) for x in text.split(",") if x.strip()]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_dir = pathlib.Path(args.save_dir) / f"{args.exp_name}-rowtop{args.row_topk}-{time.strftime('%Y%m%d-%H%M%S')}"
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"

    def log(message):
        print(message, flush=True)
        with log_path.open("a") as f:
            f.write(message + "\n")

    matrix = canonicalize_sparse_matrix(read_mtx_from_tar(args.matrix_tar))
    scale = max(1.0, float(np.max(np.abs(matrix.data))) if matrix.nnz else 1.0)
    A_scaled = (matrix / scale).astype(np.float64).tocsr()
    b = load_rhs(args.prepared_data_dir, scale, A_scaled, args.rhs_seed)
    A_scaled, b, _, reorder_stats = apply_reordering(A_scaled, b, args.reorder)
    if args.equilibrate:
        A_model, dr, dc, equil_stats = max_abs_equilibrate(
            A_scaled, args.equil_iters, args.equil_eps)
    else:
        A_model = A_scaled
        dr = np.ones(A_scaled.shape[0], dtype=np.float64)
        dc = np.ones(A_scaled.shape[0], dtype=np.float64)
        equil_stats = None

    teacher_start = time.perf_counter()
    teacher_error = None
    if args.no_teacher:
        target_csr = sp.csr_matrix(A_model.shape, dtype=np.float64)
        dense_inverse = None
        try:
            teacher = spilu(
                A_model.tocsc(),
                drop_tol=args.spilu_drop_tol,
                fill_factor=args.spilu_fill_factor,
            )
        except RuntimeError as exc:
            teacher = None
            teacher_error = str(exc)
    else:
        target_csr, teacher, dense_inverse = build_spilu_inverse_row_topk(
            A_model, args.row_topk, args.spilu_drop_tol, args.spilu_fill_factor)
    teacher_total_time = time.perf_counter() - teacher_start
    graph = build_graph_tensors(
        A_model, target_csr, device,
        feature_mode=args.feature_mode,
        target_transform=args.target_transform,
        topology_hop=args.topology_hop,
        topology_drop_tol=args.topology_drop_tol,
        topology_row_topk=args.topology_row_topk,
        spectral_pe_dim=args.spectral_pe_dim,
        target_scale_mode=args.target_scale_mode,
        base_mode=args.base_mode,
        amg_levels=args.amg_levels,
        amg_min_coarse_nodes=args.amg_min_coarse_nodes)
    recovery_values = (dc[graph["row"]] * dr[graph["col"]]).astype(np.float64, copy=False)

    log(f"device={device}")
    log(f"run_dir={run_dir}")
    log(
        f"N={A_scaled.shape[0]} A_nnz={A_scaled.nnz} support_nnz={graph['support_nnz']} "
        f"target_nnz={target_csr.nnz} topology_hop={args.topology_hop} "
        f"topology_drop_tol={args.topology_drop_tol:.3e} "
        f"topology_row_topk={args.topology_row_topk} "
        f"amg_levels={len(graph['amg_data'])} "
        f"reorder={args.reorder} spectral_pe_dim={args.spectral_pe_dim} "
        f"target_scale_mode={args.target_scale_mode} base_mode={args.base_mode} "
        f"scale={scale:.6e}")
    if graph["amg_level_shapes"]:
        log("amg_level_shapes=" + json.dumps(graph["amg_level_shapes"], indent=2))
    if reorder_stats is not None:
        log("reordering=" + json.dumps(reorder_stats, indent=2))
    if equil_stats is not None:
        log("equilibration=" + json.dumps(equil_stats, indent=2))
    inverse_absmax = None if dense_inverse is None else float(np.abs(dense_inverse).max())
    log(f"target_scale={graph['target_scale']:.6e} teacher_inverse_absmax={inverse_absmax}")
    log(f"teacher_build_total_time={teacher_total_time:.6f}s")

    diag = A_scaled.diagonal().astype(np.float64)
    safe_diag = diag.copy()
    safe_diag[safe_diag == 0.0] = 1.0
    if teacher is not None and args.equilibrate:
        spilu_apply = lambda v: dc * teacher.solve(dr * v)
    elif teacher is not None:
        spilu_apply = teacher.solve
    else:
        spilu_apply = None
    baselines = {
        "identity": run_bicgstab(A_scaled, b, lambda v: v, args.max_iter, args.rtol),
        "jacobi": run_bicgstab(A_scaled, b, lambda v: v / safe_diag, args.max_iter, args.rtol),
    }
    baselines["spilu"] = (
        None if spilu_apply is None
        else run_bicgstab(A_scaled, b, spilu_apply, args.max_iter, args.rtol)
    )
    if teacher_error is not None:
        baselines["spilu_error"] = teacher_error
    if args.no_teacher:
        baselines["target_spai"] = None
    else:
        target_result, _ = evaluate_values(
            A_scaled, b, graph["target_values"] * recovery_values, graph, args.max_iter, args.rtol)
        baselines["target_spai"] = target_result
    log("baselines=" + json.dumps(baselines, indent=2))

    model = NeuroSPAI(
        node_dim=graph["node_attr"].shape[1],
        edge_dim=graph["edge_attr"].shape[1],
        hidden_dim=args.hidden_dim,
        num_iterations=args.num_iterations,
        num_nodes=graph["node_attr"].shape[0],
        use_node_embedding=args.use_node_embedding,
        decoder_type=args.decoder_type,
        amg_levels=len(graph["amg_data"]),
    ).to(device)
    if args.load_checkpoint:
        state = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(state)
        log(f"loaded_checkpoint={args.load_checkpoint}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = 1.0 + args.weight_abs * graph["target_norm"].abs()
    damping_grid = parse_float_list(args.eval_damping_grid)
    hutchinson_gen = torch.Generator(device=device)
    hutchinson_gen.manual_seed(args.seed + 17)

    best = None
    best_epoch = -1
    latest_result = None
    for epoch in range(args.epochs + 1):
        model.train()
        pred_norm = model(
            graph["node_attr"], graph["edge_index"], graph["edge_attr"],
            graph["amg_data"])
        sq = (pred_norm - graph["target_norm"]).pow(2)
        mse_loss = (sq * weights).mean()
        alg_loss = pred_norm.new_zeros(())
        if args.hutchinson_weight > 0:
            alg_loss = hutchinson_loss(
                pred_norm, graph, args.log_output_clip,
                args.hutchinson_probes, hutchinson_gen)
        loss = args.mse_weight * mse_loss + args.hutchinson_weight * alg_loss
        mae = (pred_norm - graph["target_norm"]).abs().mean()

        if epoch > 0:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        should_eval = epoch == 0 or epoch == args.epochs or epoch % args.val_freq == 0
        if should_eval:
            model.eval()
            with torch.no_grad():
                pred_norm_eval = model(
                    graph["node_attr"], graph["edge_index"], graph["edge_attr"],
                    graph["amg_data"])
                pred_residual_t = decode_residual(
                    pred_norm_eval, graph, args.log_output_clip)
                pred_residual = pred_residual_t.detach().cpu().numpy().astype(np.float64)
                result = None
                values = None
                best_alpha = None
                for alpha in damping_grid:
                    candidate_hat_values = graph["base_values64"] + alpha * pred_residual
                    candidate_values = candidate_hat_values * recovery_values
                    candidate_result, _ = evaluate_values(
                        A_scaled, b, candidate_values, graph, args.max_iter, args.rtol)
                    if result is None or metric_key(candidate_result) < metric_key(result):
                        result = candidate_result
                        values = candidate_values
                        best_alpha = alpha
                latest_result = result
            if best is None or metric_key(result) < metric_key(best):
                best = result
                best_epoch = epoch
                torch.save(model.state_dict(), model_dir / "best_val.pt")
                np.savez_compressed(
                    model_dir / "best_values.npz",
                    row=graph["row"],
                    col=graph["col"],
                    values=values,
                    target_values=graph["target_values"] * recovery_values,
                    alpha=best_alpha,
                )
            log(
                f"epoch={epoch:04d} loss={loss.item():.6e} mae={mae.item():.6e} "
                f"mse={mse_loss.item():.6e} alg={alg_loss.item():.6e} "
                f"solver_info={result['info']} solver_iter={result['iterations']} "
                f"final_rel={result['final_rel']:.6e} alpha={best_alpha:g} best_epoch={best_epoch} "
                f"best_iter={best['iterations'] if best else -1}")
        elif epoch % args.log_freq == 0:
            log(f"epoch={epoch:04d} loss={loss.item():.6e} mae={mae.item():.6e} best_epoch={best_epoch}")

    torch.save(model.state_dict(), model_dir / "latest_model.pt")
    meta = {
        "args": vars(args),
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_result": best,
        "latest_result": latest_result,
        "baselines": baselines,
        "target_nnz": int(target_csr.nnz),
        "support_nnz": int(graph["support_nnz"]),
        "topology_drop_tol": float(args.topology_drop_tol),
        "topology_row_topk": int(args.topology_row_topk),
        "reordering": reorder_stats,
        "spectral_pe_dim": int(args.spectral_pe_dim),
        "amg_levels": len(graph["amg_data"]),
        "amg_level_shapes": graph["amg_level_shapes"],
        "target_scale_mode": args.target_scale_mode,
        "target_scale": float(graph["target_scale"]),
        "base_mode": args.base_mode,
        "equilibration": equil_stats,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log("summary=" + json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
