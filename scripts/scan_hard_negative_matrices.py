import argparse
import csv
import gzip
import io
import json
import pathlib
import sys
import tarfile
import time

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat, mmread
from scipy.sparse.linalg import bicgstab


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_neuro_spai_single import apply_reordering, max_abs_equilibrate
from utils.convert_suitesparse import canonicalize_sparse_matrix, read_mtx_from_tar


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan local matrix files for small hard-negative Krylov residual amplification.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["/mnt/h/suitesparse"],
        help="Files or directories to scan recursively.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("--max-n", type=int, default=5000)
    parser.add_argument("--max-nnz", type=int, default=300000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--rhs-mode", choices=["ones", "random", "ax_random"], default="random")
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--reorder", choices=["none", "rcm"], default="rcm")
    parser.add_argument("--equilibrate", action="store_true")
    parser.add_argument("--equil-iters", type=int, default=5)
    parser.add_argument("--equil-eps", type=float, default=1e-12)
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".tar.gz", ".mtx", ".mtx.gz", ".mat"],
    )
    return parser.parse_args()


def iter_matrix_paths(roots, extensions):
    exts = tuple(extensions)
    for root in roots:
        path = pathlib.Path(root)
        if path.is_file():
            if str(path).endswith(exts):
                yield path
            continue
        if not path.exists():
            continue
        for child in path.rglob("*"):
            if child.is_file() and str(child).endswith(exts):
                yield child


def read_matrix(path):
    text = str(path)
    if text.endswith(".tar.gz"):
        A = read_mtx_from_tar(path)
        if A is None:
            raise ValueError("no .mtx found in archive")
        return canonicalize_sparse_matrix(A)
    if text.endswith(".mtx.gz"):
        with gzip.open(path, "rb") as f:
            return canonicalize_sparse_matrix(mmread(io.BytesIO(f.read())))
    if text.endswith(".mtx"):
        return canonicalize_sparse_matrix(mmread(path))
    if text.endswith(".mat"):
        data = loadmat(path)
        if "Problem" in data:
            problem = data["Problem"]
            if problem.dtype.names and "A" in problem.dtype.names:
                return canonicalize_sparse_matrix(problem["A"][0, 0])
        for value in data.values():
            if sp.issparse(value):
                return canonicalize_sparse_matrix(value)
        raise ValueError("no sparse matrix found in .mat")
    raise ValueError(f"unsupported matrix file: {path}")


def make_rhs(A, mode, rng):
    n = A.shape[0]
    if mode == "ones":
        return np.ones(n, dtype=np.float64)
    x = rng.uniform(-1.0, 1.0, n)
    if mode == "random":
        return x.astype(np.float64, copy=False)
    if mode == "ax_random":
        return np.asarray(A @ x, dtype=np.float64).reshape(-1)
    raise ValueError(f"unknown rhs_mode={mode!r}")


def krylov_residual_ratio(A, b, steps):
    norm0 = max(1e-30, float(np.linalg.norm(b)))
    residual_history = []

    def callback(xk):
        r = b - A @ xk
        residual_history.append(float(np.linalg.norm(r) / norm0))

    try:
        x, info = bicgstab(
            A,
            b,
            x0=np.zeros(A.shape[0], dtype=np.float64),
            rtol=0.0,
            atol=0.0,
            maxiter=max(1, int(steps)),
            callback=callback,
        )
        final_rel = float(np.linalg.norm(b - A @ x) / norm0)
        if not np.isfinite(final_rel):
            raise FloatingPointError("non-finite final residual")
        status = "ok"
        error = None
    except Exception as exc:
        info = -9999
        final_rel = float("inf")
        status = "exception"
        error = str(exc)

    ratio_at_steps = residual_history[-1] if residual_history else final_rel
    hard = bool((info < 0) or (ratio_at_steps > 1.0) or (final_rel > 1.0))
    return {
        "hard": hard,
        "info": int(info),
        "ratio_at_steps": float(ratio_at_steps),
        "final_rel": float(final_rel),
        "history_len": int(len(residual_history)),
        "status": status,
        "error": error,
    }


def inspect_matrix(path, args, index):
    start = time.perf_counter()
    A = read_matrix(path)
    A = canonicalize_sparse_matrix(A)
    if A.shape[0] != A.shape[1]:
        return None, "non_square"
    n = A.shape[0]
    if n < args.min_n or n > args.max_n or A.nnz > args.max_nnz:
        return None, "size_filter"

    scale = max(1.0, float(np.max(np.abs(A.data))) if A.nnz else 1.0)
    A = (A / scale).astype(np.float64).tocsr()
    rng = np.random.default_rng(args.seed + index * 1009)
    b = make_rhs(A, args.rhs_mode, rng)
    A, b, _, reorder_stats = apply_reordering(A, b, args.reorder)
    if args.equilibrate:
        A_model, dr, _, equil_stats = max_abs_equilibrate(
            A, args.equil_iters, args.equil_eps)
        b_model = dr * b
    else:
        A_model = A
        b_model = b
        equil_stats = None

    probe = krylov_residual_ratio(A_model, b_model, args.steps)
    row = {
        "path": str(path),
        "name": path.name,
        "n": int(n),
        "nnz": int(A.nnz),
        "scale": float(scale),
        "hard": bool(probe["hard"] and probe["ratio_at_steps"] >= args.threshold),
        "ratio_at_steps": probe["ratio_at_steps"],
        "final_rel": probe["final_rel"],
        "info": probe["info"],
        "history_len": probe["history_len"],
        "status": probe["status"],
        "error": probe["error"],
        "reorder": args.reorder,
        "bandwidth_before": None if reorder_stats is None else reorder_stats["bandwidth_before"],
        "bandwidth_after": None if reorder_stats is None else reorder_stats["bandwidth_after"],
        "equilibrate": bool(args.equilibrate),
        "equil_stats": json.dumps(equil_stats),
        "scan_time": float(time.perf_counter() - start),
    }
    return row, None


def main():
    args = parse_args()
    out = pathlib.Path(args.out) if args.out else pathlib.Path(
        "results/hard_negative_scan") / f"scan_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = {}
    for idx, path in enumerate(iter_matrix_paths(args.roots, args.extensions)):
        if args.limit > 0 and idx >= args.limit:
            break
        try:
            row, skip_reason = inspect_matrix(path, args, idx)
        except Exception as exc:
            row = None
            skip_reason = f"error:{exc}"
        if row is None:
            skipped[skip_reason] = skipped.get(skip_reason, 0) + 1
            continue
        rows.append(row)
        marker = "HARD" if row["hard"] else "soft"
        print(
            f"{marker} n={row['n']} nnz={row['nnz']} "
            f"ratio={row['ratio_at_steps']:.3e} info={row['info']} {row['path']}",
            flush=True,
        )

    rows.sort(key=lambda item: item["ratio_at_steps"], reverse=True)
    fieldnames = [
        "hard", "ratio_at_steps", "final_rel", "info", "history_len", "status",
        "n", "nnz", "name", "path", "scale", "reorder", "bandwidth_before",
        "bandwidth_after", "equilibrate", "equil_stats", "scan_time", "error",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    hard_count = sum(1 for row in rows if row["hard"])
    print(json.dumps({
        "out": str(out),
        "scanned": len(rows),
        "hard": hard_count,
        "skipped": skipped,
        "top": rows[:10],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
