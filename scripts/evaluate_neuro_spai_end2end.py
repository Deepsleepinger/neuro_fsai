import argparse
import csv
import json
import pathlib
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import spilu


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_neuro_spai_single import (
    NeuroSPAI,
    apply_reordering,
    build_graph_tensors,
    csr_from_values,
    decode_residual,
    load_rhs,
    max_abs_equilibrate,
    run_bicgstab,
)
from utils.convert_suitesparse import canonicalize_sparse_matrix, read_mtx_from_tar


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end Neuro-SPAI timing with explicit preconditioner apply time.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--matrix-tars", nargs="+", required=True)
    parser.add_argument("--out-dir", default="results/end2end")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--checkpoint-name", default="best_val.pt")
    parser.add_argument("--rhs-seed", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    return parser.parse_args()


def matrix_name(path):
    return pathlib.Path(path).name.replace(".tar.gz", "")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def zero_label_matrix(shape):
    return sp.csr_matrix(shape, dtype=np.float64)


def alpha_for_matrix(meta, name, fallback):
    if fallback is not None:
        return fallback
    for section in ("best_results", "latest_results"):
        results = meta.get(section) or {}
        if name in results and results[name].get("alpha") is not None:
            return float(results[name]["alpha"])
    model_dir = pathlib.Path(meta["run_dir"]) / "model" / "best_values.npz"
    if model_dir.exists():
        values = np.load(model_dir)
        if "alpha" in values.files:
            return float(values["alpha"])
    return 1.0


def load_model(run_dir, meta, graph, device, checkpoint_name):
    args = meta["args"]
    checkpoint = pathlib.Path(run_dir) / "model" / checkpoint_name
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model = NeuroSPAI(
        node_dim=graph["node_attr"].shape[1],
        edge_dim=graph["edge_attr"].shape[1],
        hidden_dim=int(args.get("hidden_dim", 64)),
        num_iterations=int(args.get("num_iterations", 1)),
        num_nodes=graph["node_attr"].shape[0],
        use_node_embedding=bool(args.get("use_node_embedding", False)),
        decoder_type=args.get("decoder_type", "mlp"),
        amg_levels=len(graph.get("amg_data", [])),
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def align_graph_to_checkpoint(run_dir, graph, checkpoint_name):
    checkpoint = pathlib.Path(run_dir) / "model" / checkpoint_name
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected_node_dim = state.get("node_encoder.0.weight").shape[1]
    expected_edge_dim = state.get("edge_encoder.0.weight").shape[1]

    def resize_feature(tensor, expected_dim):
        current_dim = tensor.shape[1]
        if current_dim == expected_dim:
            return tensor
        if current_dim > expected_dim:
            return tensor[:, :expected_dim]
        pad = tensor.new_zeros((tensor.shape[0], expected_dim - current_dim))
        return torch.cat([tensor, pad], dim=1)

    graph = dict(graph)
    graph["node_attr"] = resize_feature(graph["node_attr"], expected_node_dim)
    graph["edge_attr"] = resize_feature(graph["edge_attr"], expected_edge_dim)
    return graph


def evaluate_matrix(run_dir, meta, tar_path, cli_args, device):
    train_args = meta["args"]
    name = matrix_name(tar_path)
    total_start = time.perf_counter()

    io_start = time.perf_counter()
    matrix = canonicalize_sparse_matrix(read_mtx_from_tar(tar_path))
    matrix_io_sec = time.perf_counter() - io_start

    scale_start = time.perf_counter()
    scale = max(1.0, float(np.max(np.abs(matrix.data))) if matrix.nnz else 1.0)
    A_scaled = (matrix / scale).astype(np.float64).tocsr()
    scaling_sec = time.perf_counter() - scale_start

    rhs_start = time.perf_counter()
    b = load_rhs(None, scale, A_scaled, cli_args.rhs_seed or train_args.get("rhs_seed", 20260622))
    rhs_sec = time.perf_counter() - rhs_start

    reorder_start = time.perf_counter()
    A_scaled, b, _, reorder_stats = apply_reordering(
        A_scaled, b, train_args.get("reorder", "none"))
    reorder_sec = time.perf_counter() - reorder_start

    equil_start = time.perf_counter()
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
    equil_sec = time.perf_counter() - equil_start
    matrix_load_sec = matrix_io_sec + scaling_sec + rhs_sec + reorder_sec + equil_sec

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
    graph = align_graph_to_checkpoint(run_dir, graph, cli_args.checkpoint_name)
    synchronize(device)
    graph_build_sec = time.perf_counter() - graph_start

    model_start = time.perf_counter()
    model = load_model(run_dir, meta, graph, device, cli_args.checkpoint_name)
    synchronize(device)
    model_load_sec = time.perf_counter() - model_start

    forward_start = time.perf_counter()
    with torch.no_grad():
        pred_norm = model(
            graph["node_attr"], graph["edge_index"], graph["edge_attr"],
            graph.get("amg_data"))
        pred_residual = decode_residual(
            pred_norm, graph, float(train_args.get("log_output_clip", 16.0)))
    synchronize(device)
    forward_sec = time.perf_counter() - forward_start

    warm_forward_start = time.perf_counter()
    with torch.no_grad():
        _ = model(
            graph["node_attr"], graph["edge_index"], graph["edge_attr"],
            graph.get("amg_data"))
    synchronize(device)
    forward_warm_sec = time.perf_counter() - warm_forward_start

    g_start = time.perf_counter()
    alpha = alpha_for_matrix(meta, name, cli_args.alpha)
    recovery_values = (dc[graph["row"]] * dr[graph["col"]]).astype(np.float64, copy=False)
    values_hat = graph["base_values64"] + alpha * pred_residual.detach().cpu().numpy().astype(np.float64)
    values = values_hat * recovery_values
    G = csr_from_values(graph["row"], graph["col"], values, A_scaled.shape)
    g_build_sec = time.perf_counter() - g_start

    max_iter = cli_args.max_iter or int(train_args.get("max_iter", 2000))
    rtol = cli_args.rtol or float(train_args.get("rtol", 1e-8))
    neuro_result = run_bicgstab(A_scaled, b, lambda v: G @ v, max_iter, rtol)

    jacobi_diag = A_scaled.diagonal().astype(np.float64)
    jacobi_diag[jacobi_diag == 0.0] = 1.0
    jacobi_result = run_bicgstab(A_scaled, b, lambda v: v / jacobi_diag, max_iter, rtol)

    spilu_start = time.perf_counter()
    teacher = spilu(
        A_model.tocsc(),
        drop_tol=float(train_args.get("spilu_drop_tol", 1e-4)),
        fill_factor=float(train_args.get("spilu_fill_factor", 10.0)),
    )
    spilu_setup_sec = time.perf_counter() - spilu_start
    if train_args.get("equilibrate", False):
        spilu_apply = lambda v: dc * teacher.solve(dr * v)
    else:
        spilu_apply = teacher.solve
    spilu_result = run_bicgstab(A_scaled, b, spilu_apply, max_iter, rtol)

    neuro_setup_no_load_sec = graph_build_sec + forward_sec + g_build_sec
    neuro_setup_warm_no_load_sec = graph_build_sec + forward_warm_sec + g_build_sec
    neuro_setup_sec = neuro_setup_no_load_sec + model_load_sec
    neuro_precompute_warm_sec = (
        scaling_sec + reorder_sec + equil_sec
        + graph_build_sec + forward_warm_sec + g_build_sec
    )
    neuro_total_sec = neuro_setup_sec + neuro_result["solve_time"]
    neuro_total_no_load_sec = neuro_setup_no_load_sec + neuro_result["solve_time"]
    neuro_total_warm_no_load_sec = neuro_setup_warm_no_load_sec + neuro_result["solve_time"]
    neuro_strict_total_warm_sec = neuro_precompute_warm_sec + neuro_result["solve_time"]
    spilu_total_sec = spilu_setup_sec + spilu_result["solve_time"]
    total_sec = time.perf_counter() - total_start

    return {
        "matrix": name,
        "n": A_scaled.shape[0],
        "a_nnz": A_scaled.nnz,
        "support_nnz": graph["support_nnz"],
        "reorder": train_args.get("reorder", "none"),
        "bandwidth_before": None if reorder_stats is None else reorder_stats["bandwidth_before"],
        "bandwidth_after": None if reorder_stats is None else reorder_stats["bandwidth_after"],
        "equilibrate": bool(train_args.get("equilibrate", False)),
        "base_mode": train_args.get("base_mode", "jacobi"),
        "equil_row_max": None if equil_stats is None else equil_stats["row_max_max"],
        "equil_col_max": None if equil_stats is None else equil_stats["col_max_max"],
        "alpha": alpha,
        "matrix_load_sec": matrix_load_sec,
        "matrix_io_sec": matrix_io_sec,
        "scaling_sec": scaling_sec,
        "rhs_sec": rhs_sec,
        "reorder_sec": reorder_sec,
        "equil_sec": equil_sec,
        "graph_build_sec": graph_build_sec,
        "model_load_sec": model_load_sec,
        "forward_sec": forward_sec,
        "forward_warm_sec": forward_warm_sec,
        "g_build_sec": g_build_sec,
        "neuro_setup_no_load_sec": neuro_setup_no_load_sec,
        "neuro_setup_warm_no_load_sec": neuro_setup_warm_no_load_sec,
        "neuro_precompute_warm_sec": neuro_precompute_warm_sec,
        "neuro_setup_sec": neuro_setup_sec,
        "neuro_solve_sec": neuro_result["solve_time"],
        "neuro_prec_sec": neuro_result["preconditioner_time"],
        "neuro_prec_calls": neuro_result["preconditioner_calls"],
        "neuro_prec_avg_sec": neuro_result["preconditioner_avg_time"],
        "neuro_total_no_load_sec": neuro_total_no_load_sec,
        "neuro_total_warm_no_load_sec": neuro_total_warm_no_load_sec,
        "neuro_strict_total_warm_sec": neuro_strict_total_warm_sec,
        "neuro_total_sec": neuro_total_sec,
        "neuro_iter": neuro_result["iterations"],
        "neuro_info": neuro_result["info"],
        "neuro_final_rel": neuro_result["final_rel"],
        "jacobi_solve_sec": jacobi_result["solve_time"],
        "jacobi_prec_sec": jacobi_result["preconditioner_time"],
        "jacobi_iter": jacobi_result["iterations"],
        "spilu_setup_sec": spilu_setup_sec,
        "spilu_solve_sec": spilu_result["solve_time"],
        "spilu_prec_sec": spilu_result["preconditioner_time"],
        "spilu_prec_calls": spilu_result["preconditioner_calls"],
        "spilu_total_sec": spilu_total_sec,
        "spilu_iter": spilu_result["iterations"],
        "script_total_sec": total_sec,
    }


def main():
    args = parse_args()
    run_dir = pathlib.Path(args.run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        _ = torch.empty((1,), device=device) + 1.0
        synchronize(device)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for tar_path in args.matrix_tars:
        row = evaluate_matrix(run_dir, meta, tar_path, args, device)
        rows.append(row)
        print(json.dumps(row), flush=True)

    csv_path = out_dir / f"end2end_{run_dir.name}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
