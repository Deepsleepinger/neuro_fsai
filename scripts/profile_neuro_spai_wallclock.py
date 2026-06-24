import argparse
import csv
import json
import pathlib
import sys
import time

import numpy as np
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
    max_abs_equilibrate,
)
from utils.convert_suitesparse import canonicalize_sparse_matrix, read_mtx_from_tar


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile Neuro-SPAI feature+forward setup time against scipy spilu setup.")
    parser.add_argument("--matrix-tars", nargs="+", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--out-dir", default="results/wallclock")
    parser.add_argument("--row-topk", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--decoder-type", choices=["mlp", "bilinear"], default="mlp")
    parser.add_argument("--feature-mode", choices=["algebraic", "legacy"], default="algebraic")
    parser.add_argument("--reorder", choices=["none", "rcm"], default="none")
    parser.add_argument("--spectral-pe-dim", type=int, default=0)
    parser.add_argument("--topology-hop", type=int, choices=[1, 2], default=1)
    parser.add_argument("--topology-drop-tol", type=float, default=0.0)
    parser.add_argument("--topology-row-topk", type=int, default=64)
    parser.add_argument("--equilibrate", action="store_true")
    parser.add_argument("--equil-iters", type=int, default=5)
    parser.add_argument("--equil-eps", type=float, default=1e-12)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--spilu-drop-tol", type=float, default=1e-4)
    parser.add_argument("--spilu-fill-factor", type=float, default=10.0)
    return parser.parse_args()


def matrix_name(path):
    return pathlib.Path(path).name.replace(".tar.gz", "")


def make_label_pattern(A_csr):
    pattern = A_csr.copy().tocsr()
    pattern.data = np.ones_like(pattern.data, dtype=np.float64)
    pattern = pattern.tolil()
    pattern.setdiag(1.0)
    pattern = pattern.tocsr()
    pattern.eliminate_zeros()
    return pattern


def load_model(args, graph, device):
    meta_args = {}
    if args.meta:
        meta_args = json.loads(pathlib.Path(args.meta).read_text()).get("args", {})
    hidden_dim = int(meta_args.get("hidden_dim", args.hidden_dim))
    num_iterations = int(meta_args.get("num_iterations", args.num_iterations))
    decoder_type = meta_args.get("decoder_type", args.decoder_type)
    use_node_embedding = bool(meta_args.get("use_node_embedding", False))
    model = NeuroSPAI(
        node_dim=graph["node_attr"].shape[1],
        edge_dim=graph["edge_attr"].shape[1],
        hidden_dim=hidden_dim,
        num_iterations=num_iterations,
        num_nodes=graph["node_attr"].shape[0],
        use_node_embedding=use_node_embedding,
        decoder_type=decoder_type,
    ).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_forward(model, graph, device, warmup, repeats):
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(graph["node_attr"], graph["edge_index"], graph["edge_attr"])
        synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(graph["node_attr"], graph["edge_index"], graph["edge_attr"])
        synchronize(device)
    return (time.perf_counter() - start) / max(1, repeats)


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        _ = torch.empty((1,), device=device) + 1.0
        synchronize(device)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for tar_path in args.matrix_tars:
        matrix = canonicalize_sparse_matrix(read_mtx_from_tar(tar_path))
        scale = max(1.0, float(np.max(np.abs(matrix.data))) if matrix.nnz else 1.0)
        A_scaled = (matrix / scale).astype(np.float64).tocsr()
        A_scaled, _, _, reorder_stats = apply_reordering(A_scaled, None, args.reorder)
        if args.equilibrate:
            A_model, _, _, _ = max_abs_equilibrate(A_scaled, args.equil_iters, args.equil_eps)
        else:
            A_model = A_scaled

        spilu_start = time.perf_counter()
        spilu(A_model.tocsc(), drop_tol=args.spilu_drop_tol, fill_factor=args.spilu_fill_factor)
        spilu_setup_sec = time.perf_counter() - spilu_start

        feature_start = time.perf_counter()
        label_csr = make_label_pattern(A_model)
        graph = build_graph_tensors(
            A_model, label_csr, device,
            feature_mode=args.feature_mode,
            topology_hop=args.topology_hop,
            topology_drop_tol=args.topology_drop_tol,
            topology_row_topk=args.topology_row_topk,
            spectral_pe_dim=args.spectral_pe_dim)
        synchronize(device)
        feature_sec = time.perf_counter() - feature_start

        model = load_model(args, graph, device)
        forward_sec = time_forward(model, graph, device, args.warmup, args.repeats)
        neuro_setup_sec = feature_sec + forward_sec
        row = {
            "matrix": matrix_name(tar_path),
            "n": A_scaled.shape[0],
            "a_nnz": A_scaled.nnz,
            "support": f"{args.topology_hop}-hop",
            "support_nnz": int(graph["support_nnz"]),
            "reorder": args.reorder,
            "bandwidth_before": None if reorder_stats is None else reorder_stats["bandwidth_before"],
            "bandwidth_after": None if reorder_stats is None else reorder_stats["bandwidth_after"],
            "spectral_pe_dim": int(args.spectral_pe_dim),
            "feature_sec": feature_sec,
            "forward_sec": forward_sec,
            "neuro_setup_sec": neuro_setup_sec,
            "spilu_setup_sec": spilu_setup_sec,
            "neuro_over_spilu": neuro_setup_sec / max(spilu_setup_sec, 1e-12),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    csv_path = out_dir / f"wallclock_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
