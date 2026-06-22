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
        description="Benchmark sparse truncations of an explicit SPILU inverse.")
    parser.add_argument("--matrix-tar", required=True)
    parser.add_argument(
        "--prepared-data-dir",
        required=True,
        help="Directory containing train/train_0000.npy; only its RHS is used.")
    parser.add_argument("--output", default="results/spilu_inverse_topk_single.json")
    parser.add_argument("--spilu-drop-tol", type=float, default=1e-4)
    parser.add_argument("--spilu-fill-factor", type=float, default=10.0)
    parser.add_argument("--global-topk-list", default="32632,132632,300000,600000")
    parser.add_argument("--row-topk-list", default="16,32,64,112,256,512")
    parser.add_argument("--max-dense-n", type=int, default=5000)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--rtol", type=float, default=1e-8)
    return parser.parse_args()


def parse_int_list(text):
    return [int(x) for x in text.split(",") if x.strip()]


def load_rhs(prepared_data_dir, scale):
    path = pathlib.Path(prepared_data_dir) / "train" / "train_0000.npy"
    data = np.load(path, allow_pickle=True)
    if len(data) != 1:
        raise ValueError(f"Expected one graph in {path}, got {len(data)}")
    return np.asarray(data[0]["rhs"], dtype=np.float64).reshape(-1) / scale


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


def sparse_global_topk(dense_inverse, topk):
    n = dense_inverse.shape[0]
    flat_abs = np.abs(dense_inverse).ravel()
    if topk >= flat_abs.size:
        selected = np.ones(flat_abs.size, dtype=bool)
    else:
        idx = np.argpartition(flat_abs, -topk)[-topk:]
        selected = np.zeros(flat_abs.size, dtype=bool)
        selected[idx] = True
    rows, cols = np.nonzero(selected.reshape(n, n))
    vals = dense_inverse[rows, cols]
    G = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    G.eliminate_zeros()
    return G


def sparse_row_topk(dense_inverse, topk):
    n = dense_inverse.shape[0]
    topk = min(topk, n)
    rows = []
    cols = []
    vals = []
    for row in range(n):
        row_abs = np.abs(dense_inverse[row])
        idx = np.argpartition(row_abs, -topk)[-topk:]
        rows.append(np.full(topk, row, dtype=np.int64))
        cols.append(idx.astype(np.int64, copy=False))
        vals.append(dense_inverse[row, idx])
    G = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()
    G.eliminate_zeros()
    return G


def print_table(results):
    print("method,nnz,info,iterations,solve_time,final_rel", flush=True)
    for name, row in results.items():
        print(
            f"{name},{row.get('nnz', '')},{row['info']},{row['iterations']},"
            f"{row['solve_time']:.6f},{row['final_rel']:.6e}",
            flush=True,
        )


def main():
    args = parse_args()
    matrix = canonicalize_sparse_matrix(read_mtx_from_tar(args.matrix_tar))
    if matrix.shape[0] > args.max_dense_n:
        raise ValueError(
            f"N={matrix.shape[0]} exceeds --max-dense-n={args.max_dense_n}; "
            "explicit inverse top-k is a small/medium diagnostic only.")

    scale = max(1.0, float(np.max(np.abs(matrix.data))) if matrix.nnz else 1.0)
    A_scaled = (matrix / scale).astype(np.float64).tocsr()
    b = load_rhs(args.prepared_data_dir, scale)
    diag = A_scaled.diagonal().astype(np.float64)
    diag[diag == 0.0] = 1.0

    teacher_start = time.perf_counter()
    teacher = spilu(
        A_scaled.tocsc(),
        drop_tol=args.spilu_drop_tol,
        fill_factor=args.spilu_fill_factor,
    )
    teacher_setup_time = time.perf_counter() - teacher_start

    inverse_start = time.perf_counter()
    dense_inverse = np.asarray(teacher.solve(np.eye(A_scaled.shape[0])), dtype=np.float64)
    inverse_build_time = time.perf_counter() - inverse_start

    results = {
        "identity": run_bicgstab(A_scaled, b, lambda v: v, args.max_iter, args.rtol),
        "jacobi": run_bicgstab(A_scaled, b, lambda v: v / diag, args.max_iter, args.rtol),
        "spilu": run_bicgstab(A_scaled, b, teacher.solve, args.max_iter, args.rtol),
    }
    results["identity"]["nnz"] = int(A_scaled.shape[0])
    results["jacobi"]["nnz"] = int(A_scaled.shape[0])
    results["spilu"]["nnz"] = int(teacher.L.nnz + teacher.U.nnz)

    for topk in parse_int_list(args.global_topk_list):
        G = sparse_global_topk(dense_inverse, topk)
        key = f"denseinv_global_top{topk}"
        results[key] = run_bicgstab(A_scaled, b, lambda v, mat=G: mat @ v, args.max_iter, args.rtol)
        results[key]["nnz"] = int(G.nnz)

    for topk in parse_int_list(args.row_topk_list):
        G = sparse_row_topk(dense_inverse, topk)
        key = f"denseinv_row_top{topk}"
        results[key] = run_bicgstab(A_scaled, b, lambda v, mat=G: mat @ v, args.max_iter, args.rtol)
        results[key]["nnz"] = int(G.nnz)

    print_table(results)

    report = {
        "matrix_tar": args.matrix_tar,
        "prepared_data_dir": args.prepared_data_dir,
        "N": int(A_scaled.shape[0]),
        "nnz": int(A_scaled.nnz),
        "scale": float(scale),
        "spilu_drop_tol": float(args.spilu_drop_tol),
        "spilu_fill_factor": float(args.spilu_fill_factor),
        "teacher_setup_time": float(teacher_setup_time),
        "inverse_build_time": float(inverse_build_time),
        "dense_inverse_absmax": float(np.max(np.abs(dense_inverse))),
        "dense_inverse_mean_abs": float(np.mean(np.abs(dense_inverse))),
        "results": results,
    }
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"saved={output_path}", flush=True)


if __name__ == "__main__":
    main()
