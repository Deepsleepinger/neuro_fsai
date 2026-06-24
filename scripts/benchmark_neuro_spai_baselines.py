import argparse
import contextlib
import csv
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
from scipy.sparse.linalg import spilu

try:
    import pyamg
except Exception:  # pragma: no cover - dependency may be absent on some machines.
    pyamg = None


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_neuro_spai_end2end import (  # noqa: E402
    align_graph_to_checkpoint,
    alpha_for_matrix,
    load_model,
    synchronize,
    zero_label_matrix,
)
from train_neuro_spai_single import (  # noqa: E402
    apply_reordering,
    build_graph_tensors,
    csr_from_values,
    decode_residual,
    load_rhs,
    max_abs_equilibrate,
    run_bicgstab,
)
from utils.convert_suitesparse import canonicalize_sparse_matrix, read_mtx_from_tar  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Neuro-SPAI against Jacobi, PyAMG and SciPy ILU on multiple matrices. "
            "Use --case MATRIX_TAR[::RUN_DIR[::CHECKPOINT]]; omit RUN_DIR for baselines only."
        )
    )
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--out-dir", default="results/baseline_benchmarks")
    parser.add_argument("--rhs-seed", type=int, default=20260622)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint-name", default="best_val.pt")
    parser.add_argument("--reorder", choices=["none", "rcm"], default="rcm")
    parser.add_argument("--equilibrate", action="store_true")
    parser.add_argument("--equil-iters", type=int, default=5)
    parser.add_argument("--equil-eps", type=float, default=1e-12)
    parser.add_argument("--spilu-drop-tol", type=float, default=1e-4)
    parser.add_argument("--spilu-fill-factor", type=float, default=10.0)
    parser.add_argument(
        "--amg-method",
        choices=["smoothed_aggregation", "ruge_stuben"],
        default="smoothed_aggregation",
    )
    parser.add_argument("--amg-cycle", default="V")
    parser.add_argument("--amg-max-coarse", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=None)
    return parser.parse_args()


def matrix_name(path):
    return pathlib.Path(path).name.replace(".tar.gz", "")


def parse_case(case_text, default_checkpoint):
    parts = case_text.split("::")
    if len(parts) > 3:
        raise ValueError(f"bad case format: {case_text!r}")
    matrix_tar = parts[0]
    run_dir = parts[1] if len(parts) >= 2 and parts[1] else None
    checkpoint = parts[2] if len(parts) >= 3 and parts[2] else default_checkpoint
    return matrix_tar, run_dir, checkpoint


def choose_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def success_status(result):
    if result.get("status") == "error":
        return "error"
    return "ok" if int(result.get("info", 1)) == 0 else "not_converged"


def solver_error(error, setup_sec=0.0):
    return {
        "status": "error",
        "iterations": None,
        "raw_callback_iterations": None,
        "solve_time": None,
        "preconditioner_time": None,
        "preconditioner_calls": None,
        "preconditioner_avg_time": None,
        "info": None,
        "final_rel": None,
        "setup_sec": float(setup_sec),
        "error": f"{type(error).__name__}: {error}",
    }


def run_solver_safe(A_csr, b, apply_prec, max_iter, rtol):
    try:
        result = run_bicgstab(A_csr, b, apply_prec, max_iter, rtol)
        result["status"] = success_status(result)
        result["setup_sec"] = 0.0
        result["error"] = ""
        return result
    except Exception as error:
        return solver_error(error)


def flatten_result(prefix, result):
    result = result or {}
    solve_sec = result.get("solve_time")
    setup_sec = result.get("setup_sec")
    total_sec = None
    if solve_sec is not None and setup_sec is not None:
        total_sec = float(solve_sec) + float(setup_sec)
    return {
        f"{prefix}_status": result.get("status"),
        f"{prefix}_iter": result.get("iterations"),
        f"{prefix}_raw_iter": result.get("raw_callback_iterations"),
        f"{prefix}_info": result.get("info"),
        f"{prefix}_final_rel": result.get("final_rel"),
        f"{prefix}_setup_sec": setup_sec,
        f"{prefix}_solve_sec": solve_sec,
        f"{prefix}_total_sec": total_sec,
        f"{prefix}_prec_sec": result.get("preconditioner_time"),
        f"{prefix}_prec_calls": result.get("preconditioner_calls"),
        f"{prefix}_error": result.get("error", ""),
    }


def load_meta(run_dir):
    if run_dir is None:
        return None
    meta_path = pathlib.Path(run_dir) / "meta.json"
    return json.loads(meta_path.read_text())


def default_train_args(args):
    return {
        "reorder": args.reorder,
        "equilibrate": bool(args.equilibrate),
        "equil_iters": args.equil_iters,
        "equil_eps": args.equil_eps,
        "spilu_drop_tol": args.spilu_drop_tol,
        "spilu_fill_factor": args.spilu_fill_factor,
        "rhs_seed": args.rhs_seed,
    }


def preprocess_matrix(tar_path, train_args, rhs_seed):
    matrix = canonicalize_sparse_matrix(read_mtx_from_tar(tar_path))
    scale = max(1.0, float(np.max(np.abs(matrix.data))) if matrix.nnz else 1.0)
    A_scaled = (matrix / scale).astype(np.float64).tocsr()
    b = load_rhs(None, scale, A_scaled, rhs_seed)
    A_scaled, b, _, reorder_stats = apply_reordering(
        A_scaled, b, train_args.get("reorder", "none"))
    if train_args.get("equilibrate", False):
        A_model, dr, dc, equil_stats = max_abs_equilibrate(
            A_scaled,
            int(train_args.get("equil_iters", 5)),
            float(train_args.get("equil_eps", 1e-12)),
        )
    else:
        A_model = A_scaled
        dr = np.ones(A_scaled.shape[0], dtype=np.float64)
        dc = np.ones(A_scaled.shape[0], dtype=np.float64)
        equil_stats = None
    return matrix, A_scaled, A_model, b, dr, dc, reorder_stats, equil_stats


def evaluate_jacobi(A_scaled, b, max_iter, rtol):
    diag = A_scaled.diagonal().astype(np.float64)
    diag[diag == 0.0] = 1.0
    return run_solver_safe(A_scaled, b, lambda v: v / diag, max_iter, rtol)


def evaluate_ilu(A_scaled, A_model, b, dr, dc, train_args, max_iter, rtol):
    start = time.perf_counter()
    try:
        ilu = spilu(
            A_model.tocsc(),
            drop_tol=float(train_args.get("spilu_drop_tol", 1e-4)),
            fill_factor=float(train_args.get("spilu_fill_factor", 10.0)),
        )
        setup_sec = time.perf_counter() - start
        result = run_solver_safe(A_scaled, b, lambda v: dc * ilu.solve(dr * v), max_iter, rtol)
        result["setup_sec"] = setup_sec
        return result
    except Exception as error:
        return solver_error(error, time.perf_counter() - start)


def build_amg_solver(A_model, args):
    if pyamg is None:
        raise RuntimeError("pyamg is not installed")
    if args.amg_method == "smoothed_aggregation":
        return pyamg.smoothed_aggregation_solver(A_model, max_coarse=args.amg_max_coarse)
    return pyamg.ruge_stuben_solver(A_model, max_coarse=args.amg_max_coarse)


def evaluate_amg(A_scaled, A_model, b, dr, dc, args):
    start = time.perf_counter()
    try:
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                ml = build_amg_solver(A_model, args)
        M = ml.aspreconditioner(cycle=args.amg_cycle)
        setup_sec = time.perf_counter() - start
        result = run_solver_safe(
            A_scaled, b, lambda v: dc * M.matvec(dr * v), args.max_iter, args.rtol)
        result["setup_sec"] = setup_sec
        return result
    except Exception as error:
        return solver_error(error, time.perf_counter() - start)


def evaluate_neuro(run_dir, checkpoint_name, meta, name, A_scaled, A_model, b, dr, dc, args, device):
    if run_dir is None:
        return {"status": "skipped", "error": "no run_dir"}
    train_args = meta.get("args", {})
    start = time.perf_counter()
    try:
        graph_start = time.perf_counter()
        graph = build_graph_tensors(
            A_model,
            zero_label_matrix(A_model.shape),
            device,
            feature_mode=train_args.get("feature_mode", "algebraic"),
            target_transform=train_args.get("target_transform", "linear"),
            topology_hop=int(train_args.get("topology_hop", 1)),
            topology_drop_tol=float(train_args.get("topology_drop_tol", 0.0)),
            topology_row_topk=int(train_args.get("topology_row_topk", 64)),
            spectral_pe_dim=int(train_args.get("spectral_pe_dim", 0)),
            target_scale_mode=train_args.get("target_scale_mode", "teacher"),
            base_mode=train_args.get("base_mode", "jacobi"),
            amg_levels=int(train_args.get("amg_levels", 0)),
            amg_min_coarse_nodes=int(train_args.get("amg_min_coarse_nodes", 500)),
        )
        graph = align_graph_to_checkpoint(run_dir, graph, checkpoint_name)
        synchronize(device)
        graph_build_sec = time.perf_counter() - graph_start

        model_start = time.perf_counter()
        model = load_model(run_dir, meta, graph, device, checkpoint_name)
        synchronize(device)
        model_load_sec = time.perf_counter() - model_start

        forward_start = time.perf_counter()
        with torch.no_grad():
            pred_norm = model(
                graph["node_attr"],
                graph["edge_index"],
                graph["edge_attr"],
                graph.get("amg_data"),
            )
            pred_residual = decode_residual(
                pred_norm, graph, float(train_args.get("log_output_clip", 16.0)))
        synchronize(device)
        forward_sec = time.perf_counter() - forward_start

        g_start = time.perf_counter()
        alpha = alpha_for_matrix(meta, name, args.alpha)
        recovery_values = (dc[graph["row"]] * dr[graph["col"]]).astype(np.float64, copy=False)
        values_hat = (
            graph["base_values64"]
            + alpha * pred_residual.detach().cpu().numpy().astype(np.float64)
        )
        values = values_hat * recovery_values
        G = csr_from_values(graph["row"], graph["col"], values, A_scaled.shape)
        g_build_sec = time.perf_counter() - g_start

        result = run_solver_safe(A_scaled, b, lambda v: G @ v, args.max_iter, args.rtol)
        result["setup_sec"] = graph_build_sec + model_load_sec + forward_sec + g_build_sec
        result["graph_build_sec"] = graph_build_sec
        result["model_load_sec"] = model_load_sec
        result["forward_sec"] = forward_sec
        result["g_build_sec"] = g_build_sec
        result["support_nnz"] = int(graph["support_nnz"])
        result["alpha"] = float(alpha)
        return result
    except Exception as error:
        result = solver_error(error, time.perf_counter() - start)
        result["graph_build_sec"] = None
        result["model_load_sec"] = None
        result["forward_sec"] = None
        result["g_build_sec"] = None
        result["support_nnz"] = None
        result["alpha"] = None
        return result


def evaluate_case(case_text, args, device):
    tar_path, run_dir, checkpoint_name = parse_case(case_text, args.checkpoint_name)
    meta = load_meta(run_dir)
    train_args = meta.get("args", {}) if meta else default_train_args(args)
    name = matrix_name(tar_path)
    rhs_seed = int(train_args.get("rhs_seed", args.rhs_seed))
    matrix, A_scaled, A_model, b, dr, dc, reorder_stats, equil_stats = preprocess_matrix(
        tar_path, train_args, rhs_seed)

    row = {
        "matrix": name,
        "matrix_tar": tar_path,
        "n": int(A_scaled.shape[0]),
        "a_nnz": int(A_scaled.nnz),
        "run_dir": run_dir or "",
        "checkpoint": checkpoint_name if run_dir else "",
        "reorder": train_args.get("reorder", "none"),
        "bandwidth_before": None if reorder_stats is None else reorder_stats["bandwidth_before"],
        "bandwidth_after": None if reorder_stats is None else reorder_stats["bandwidth_after"],
        "equilibrate": bool(train_args.get("equilibrate", False)),
        "equil_row_max": None if equil_stats is None else equil_stats["row_max_max"],
        "equil_col_max": None if equil_stats is None else equil_stats["col_max_max"],
        "max_iter": int(args.max_iter),
        "rtol": float(args.rtol),
        "amg_method": args.amg_method,
        "ilu_drop_tol": float(train_args.get("spilu_drop_tol", args.spilu_drop_tol)),
        "ilu_fill_factor": float(train_args.get("spilu_fill_factor", args.spilu_fill_factor)),
    }

    neuro = evaluate_neuro(run_dir, checkpoint_name, meta, name, A_scaled, A_model, b, dr, dc, args, device)
    jacobi = evaluate_jacobi(A_scaled, b, args.max_iter, args.rtol)
    amg = evaluate_amg(A_scaled, A_model, b, dr, dc, args)
    ilu = evaluate_ilu(A_scaled, A_model, b, dr, dc, train_args, args.max_iter, args.rtol)

    row.update(flatten_result("neuro", neuro))
    row.update({
        "neuro_graph_build_sec": neuro.get("graph_build_sec"),
        "neuro_model_load_sec": neuro.get("model_load_sec"),
        "neuro_forward_sec": neuro.get("forward_sec"),
        "neuro_g_build_sec": neuro.get("g_build_sec"),
        "neuro_support_nnz": neuro.get("support_nnz"),
        "neuro_alpha": neuro.get("alpha"),
    })
    row.update(flatten_result("jacobi", jacobi))
    row.update(flatten_result("amg", amg))
    row.update(flatten_result("ilu", ilu))
    return row


def main():
    args = parse_args()
    device = choose_device(args.device)
    if device.type == "cuda":
        _ = torch.empty((1,), device=device) + 1.0
        synchronize(device)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_text in args.case:
        row = evaluate_case(case_text, args, device)
        rows.append(row)
        print(json.dumps({
            "matrix": row["matrix"],
            "neuro": [row.get("neuro_status"), row.get("neuro_iter"), row.get("neuro_info")],
            "jacobi": [row.get("jacobi_status"), row.get("jacobi_iter"), row.get("jacobi_info")],
            "amg": [row.get("amg_status"), row.get("amg_iter"), row.get("amg_info")],
            "ilu": [row.get("ilu_status"), row.get("ilu_iter"), row.get("ilu_info")],
        }), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = out_dir / f"baseline_benchmark_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
