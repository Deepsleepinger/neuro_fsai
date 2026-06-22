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
from scipy.sparse.linalg import LinearOperator, bicgstab, spilu


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model_neuro_fsai import PDEDirectedConv
from utils.convert_suitesparse import canonicalize_sparse_matrix, read_mtx_from_tar, values_on_pattern


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
    parser.add_argument("--row-topk", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-iterations", type=int, default=2)
    parser.add_argument("--use-node-embedding", action="store_true")
    parser.add_argument("--weight-abs", type=float, default=5.0)
    parser.add_argument("--eval-damping-grid", default="1.0,0.5,0.25,0.1,0.05,0.01")
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
                 num_nodes=None, use_node_embedding=False):
        super().__init__()
        self.use_node_embedding = use_node_embedding
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
        self.edge_decoder = make_mlp(3 * hidden_dim, 1, hidden_dim, 2, norm=False)
        self._zero_init_last()

    def _zero_init_last(self):
        for layer in reversed(self.edge_decoder):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                return

    def forward(self, node_attr, edge_index, edge_attr):
        node_feat = self.node_encoder(node_attr)
        if self.node_embedding is not None:
            node_ids = torch.arange(node_attr.shape[0], device=node_attr.device)
            node_feat = node_feat + self.node_embedding(node_ids)
        edge_feat = self.edge_encoder(edge_attr)
        x = node_feat
        for mp in self.mp_layers:
            x = torch.relu(mp(x, edge_index, edge_feat))
        source, target = edge_index
        decoded = self.edge_decoder(torch.cat([x[source], x[target], edge_feat], dim=-1))
        return decoded.squeeze(-1)


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


def build_graph_tensors(A_scaled, target_csr, device):
    n = A_scaled.shape[0]
    target_coo = target_csr.tocoo()
    row = target_coo.row.astype(np.int64, copy=False)
    col = target_coo.col.astype(np.int64, copy=False)
    target_values = target_coo.data.astype(np.float64, copy=False)
    A_values = values_on_pattern(A_scaled, row, col)
    diag = A_scaled.diagonal().astype(np.float64)
    safe_diag = diag.copy()
    safe_diag[safe_diag == 0.0] = 1.0
    base_values = np.zeros_like(target_values)
    diag_mask = row == col
    base_values[diag_mask] = 1.0 / safe_diag[row[diag_mask]]
    residual_values = target_values - base_values
    target_scale = max(1e-12, float(np.max(np.abs(residual_values))))
    target_norm = residual_values / target_scale

    row_degree = np.diff(target_csr.indptr).astype(np.float32)
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
    edge_index = np.stack([col, row], axis=0).astype(np.int64)

    return {
        "node_attr": torch.from_numpy(node_attr).to(device),
        "edge_attr": torch.from_numpy(edge_attr).to(device),
        "edge_index": torch.from_numpy(edge_index).to(device),
        "target_norm": torch.from_numpy(target_norm.astype(np.float32)).to(device),
        "base_values": torch.from_numpy(base_values.astype(np.float32)).to(device),
        "base_values64": base_values.astype(np.float64, copy=False),
        "target_scale": float(target_scale),
        "row": row,
        "col": col,
        "target_values": target_values,
        "base_values_np": base_values,
        "safe_diag": safe_diag,
    }


def csr_from_values(row, col, values, shape):
    G = sp.coo_matrix((values.astype(np.float64, copy=False), (row, col)), shape=shape).tocsr()
    G.sum_duplicates()
    G.eliminate_zeros()
    return G


def run_bicgstab(A_csr, b, apply_prec, max_iter, rtol):
    counter = {"iters": 0}
    norm0 = max(1e-30, float(np.linalg.norm(b)))

    def callback(_xk):
        counter["iters"] += 1

    op = LinearOperator(A_csr.shape, matvec=apply_prec, dtype=np.float64)
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
        "info": int(info),
        "final_rel": final_rel,
    }


def evaluate_values(A_scaled, b, values, graph, max_iter, rtol):
    G = csr_from_values(graph["row"], graph["col"], values, A_scaled.shape)
    return run_bicgstab(A_scaled, b, lambda v: G @ v, max_iter, rtol), G


def metric_key(result):
    failed = 1 if result["info"] != 0 else 0
    return (failed, result["iterations"], result["final_rel"])


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

    teacher_start = time.perf_counter()
    target_csr, teacher, dense_inverse = build_spilu_inverse_row_topk(
        A_scaled, args.row_topk, args.spilu_drop_tol, args.spilu_fill_factor)
    teacher_total_time = time.perf_counter() - teacher_start
    graph = build_graph_tensors(A_scaled, target_csr, device)

    log(f"device={device}")
    log(f"run_dir={run_dir}")
    log(f"N={A_scaled.shape[0]} A_nnz={A_scaled.nnz} target_nnz={target_csr.nnz} scale={scale:.6e}")
    log(f"target_scale={graph['target_scale']:.6e} teacher_inverse_absmax={np.abs(dense_inverse).max():.6e}")
    log(f"teacher_build_total_time={teacher_total_time:.6f}s")

    diag = graph["safe_diag"]
    baselines = {
        "identity": run_bicgstab(A_scaled, b, lambda v: v, args.max_iter, args.rtol),
        "jacobi": run_bicgstab(A_scaled, b, lambda v: v / diag, args.max_iter, args.rtol),
        "spilu": run_bicgstab(A_scaled, b, teacher.solve, args.max_iter, args.rtol),
    }
    target_result, _ = evaluate_values(
        A_scaled, b, graph["target_values"], graph, args.max_iter, args.rtol)
    baselines["target_spai"] = target_result
    log("baselines=" + json.dumps(baselines, indent=2))

    model = NeuroSPAI(
        node_dim=graph["node_attr"].shape[1],
        edge_dim=graph["edge_attr"].shape[1],
        hidden_dim=args.hidden_dim,
        num_iterations=args.num_iterations,
        num_nodes=graph["node_attr"].shape[0],
        use_node_embedding=args.use_node_embedding,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = 1.0 + args.weight_abs * graph["target_norm"].abs()
    damping_grid = parse_float_list(args.eval_damping_grid)

    best = None
    best_epoch = -1
    latest_result = None
    for epoch in range(args.epochs + 1):
        model.train()
        pred_norm = model(graph["node_attr"], graph["edge_index"], graph["edge_attr"])
        sq = (pred_norm - graph["target_norm"]).pow(2)
        loss = (sq * weights).mean()
        mae = (pred_norm - graph["target_norm"]).abs().mean()

        if epoch > 0:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        should_eval = epoch == 0 or epoch == args.epochs or epoch % args.val_freq == 0
        if should_eval:
            model.eval()
            with torch.no_grad():
                pred_norm_eval = model(graph["node_attr"], graph["edge_index"], graph["edge_attr"])
                pred_residual = pred_norm_eval.detach().cpu().numpy().astype(np.float64) * graph["target_scale"]
                result = None
                values = None
                best_alpha = None
                for alpha in damping_grid:
                    candidate_values = graph["base_values64"] + alpha * pred_residual
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
                    target_values=graph["target_values"],
                    alpha=best_alpha,
                )
            log(
                f"epoch={epoch:04d} loss={loss.item():.6e} mae={mae.item():.6e} "
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
        "target_scale": float(graph["target_scale"]),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log("summary=" + json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
