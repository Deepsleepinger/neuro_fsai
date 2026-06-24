import argparse
import json
import pathlib
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, bicgstab, spilu


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.convert_suitesparse import canonicalize_sparse_matrix, read_mtx_from_tar


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a row-wise sparse inverse oracle by imitating SPILU actions.")
    parser.add_argument("--matrix-tar", required=True)
    parser.add_argument(
        "--prepared-data-dir",
        required=True,
        help="Directory containing train/train_0000.npy from the candidate topology dataset.")
    parser.add_argument("--output", default="results/spilu_action_oracle_single.json")
    parser.add_argument("--save-matrix", default=None)
    parser.add_argument("--probes", type=int, default=256)
    parser.add_argument("--val-probes", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--spilu-drop-tol", type=float, default=1e-4)
    parser.add_argument("--spilu-fill-factor", type=float, default=10.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument(
        "--damping-grid",
        default="1.0,0.5,0.25,0.1",
        help="Comma-separated alpha values for Jacobi + alpha * (G_oracle - Jacobi).")
    return parser.parse_args()


def load_single_graph(prepared_data_dir):
    path = pathlib.Path(prepared_data_dir) / "train" / "train_0000.npy"
    data = np.load(path, allow_pickle=True)
    if len(data) != 1:
        raise ValueError(f"Expected one graph in {path}, got {len(data)}")
    return data[0]


def candidate_pattern_from_graph(graph, num_nodes):
    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    rows = edge_index[1]
    cols = edge_index[0]
    values = np.ones(rows.shape[0], dtype=np.float64)
    pattern = sp.coo_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    pattern.sum_duplicates()
    pattern = (pattern != 0).astype(np.float64).tocsr()
    pattern = ((pattern != 0) + sp.eye(num_nodes, format="csr", dtype=bool)).astype(np.float64).tocsr()
    return pattern


def rademacher(num_nodes, num_probes, rng):
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(num_nodes, num_probes))


def solve_ridge_row(X, y, ridge):
    # X: [num_probes, row_support], y: [num_probes].
    num_probes, support = X.shape
    if support == 0:
        return np.empty(0, dtype=np.float64)
    if support <= num_probes:
        lhs = X.T @ X
        lhs.flat[:: support + 1] += ridge
        rhs = X.T @ y
        return np.linalg.solve(lhs, rhs)
    lhs = X @ X.T
    lhs.flat[:: num_probes + 1] += ridge
    dual = np.linalg.solve(lhs, y)
    return X.T @ dual


def build_oracle_inverse(pattern_csr, probes, teacher_actions, ridge):
    rows = []
    cols = []
    vals = []
    indptr = pattern_csr.indptr
    indices = pattern_csr.indices
    start = time.perf_counter()
    for row in range(pattern_csr.shape[0]):
        support_cols = indices[indptr[row]:indptr[row + 1]]
        X = probes[support_cols, :].T
        y = teacher_actions[row, :]
        weights = solve_ridge_row(X, y, ridge)
        if weights.size:
            keep = np.abs(weights) > 0.0
            if keep.any():
                rows.append(np.full(int(keep.sum()), row, dtype=np.int64))
                cols.append(support_cols[keep].astype(np.int64, copy=False))
                vals.append(weights[keep])
    if rows:
        row_idx = np.concatenate(rows)
        col_idx = np.concatenate(cols)
        data = np.concatenate(vals).astype(np.float64, copy=False)
    else:
        row_idx = np.empty(0, dtype=np.int64)
        col_idx = np.empty(0, dtype=np.int64)
        data = np.empty(0, dtype=np.float64)
    G = sp.coo_matrix((data, (row_idx, col_idx)), shape=pattern_csr.shape).tocsr()
    G.sum_duplicates()
    G.eliminate_zeros()
    return G, time.perf_counter() - start


def relative_action_error(G, probes, target):
    pred = G @ probes
    denom = max(1e-30, float(np.linalg.norm(target)))
    return float(np.linalg.norm(pred - target) / denom)


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


def benchmark(A_csr, b, G, teacher, max_iter, rtol, damping_grid):
    diag = A_csr.diagonal().astype(np.float64)
    safe_diag = diag.copy()
    safe_diag[safe_diag == 0.0] = 1.0
    J = sp.diags(1.0 / safe_diag, format="csr")

    results = {
        "identity": run_bicgstab(A_csr, b, lambda v: v, max_iter, rtol),
        "jacobi": run_bicgstab(A_csr, b, lambda v: v / safe_diag, max_iter, rtol),
        "spilu": run_bicgstab(A_csr, b, teacher.solve, max_iter, rtol),
        "oracle": run_bicgstab(A_csr, b, lambda v: G @ v, max_iter, rtol),
    }
    for alpha in damping_grid:
        G_alpha = J + float(alpha) * (G - J)
        results[f"oracle_damped_{alpha:g}"] = run_bicgstab(
            A_csr, b, lambda v, mat=G_alpha: mat @ v, max_iter, rtol)
    return results


def print_table(results):
    print("method,info,iterations,solve_time,final_rel", flush=True)
    for name, row in results.items():
        print(
            f"{name},{row['info']},{row['iterations']},"
            f"{row['solve_time']:.6f},{row['final_rel']:.6e}",
            flush=True,
        )


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    matrix = canonicalize_sparse_matrix(read_mtx_from_tar(args.matrix_tar))
    scale = max(1.0, float(np.max(np.abs(matrix.data))) if matrix.nnz else 1.0)
    A_scaled = (matrix / scale).astype(np.float64).tocsr()
    graph = load_single_graph(args.prepared_data_dir)
    pattern = candidate_pattern_from_graph(graph, A_scaled.shape[0])

    print(
        f"matrix={args.matrix_tar} N={A_scaled.shape[0]} nnz={A_scaled.nnz} "
        f"scale={scale:.6e} pattern_nnz={pattern.nnz}",
        flush=True,
    )

    teacher_start = time.perf_counter()
    teacher = spilu(
        A_scaled.tocsc(),
        drop_tol=args.spilu_drop_tol,
        fill_factor=args.spilu_fill_factor,
    )
    teacher_setup_time = time.perf_counter() - teacher_start

    probes = rademacher(A_scaled.shape[0], args.probes, rng)
    teacher_actions = np.asarray(teacher.solve(probes), dtype=np.float64)
    G, oracle_build_time = build_oracle_inverse(pattern, probes, teacher_actions, args.ridge)

    val_probes = rademacher(A_scaled.shape[0], args.val_probes, rng)
    val_teacher_actions = np.asarray(teacher.solve(val_probes), dtype=np.float64)
    train_action_rel_error = relative_action_error(G, probes, teacher_actions)
    val_action_rel_error = relative_action_error(G, val_probes, val_teacher_actions)

    rhs = np.asarray(graph["rhs"], dtype=np.float64).reshape(-1) / scale
    damping_grid = [float(x) for x in args.damping_grid.split(",") if x.strip()]
    results = benchmark(A_scaled, rhs, G, teacher, args.max_iter, args.rtol, damping_grid)
    print_table(results)

    if args.save_matrix:
        save_path = pathlib.Path(args.save_matrix)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        sp.save_npz(save_path, G)

    report = {
        "matrix_tar": args.matrix_tar,
        "prepared_data_dir": args.prepared_data_dir,
        "N": int(A_scaled.shape[0]),
        "nnz": int(A_scaled.nnz),
        "scale": float(scale),
        "pattern_nnz": int(pattern.nnz),
        "oracle_nnz": int(G.nnz),
        "probes": int(args.probes),
        "val_probes": int(args.val_probes),
        "ridge": float(args.ridge),
        "spilu_drop_tol": float(args.spilu_drop_tol),
        "spilu_fill_factor": float(args.spilu_fill_factor),
        "teacher_setup_time": float(teacher_setup_time),
        "oracle_build_time": float(oracle_build_time),
        "train_action_rel_error": train_action_rel_error,
        "val_action_rel_error": val_action_rel_error,
        "results": results,
    }
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"saved={output_path}", flush=True)


if __name__ == "__main__":
    main()
