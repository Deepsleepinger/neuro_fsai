import argparse
import io
import json
import pathlib
import sys
import tarfile
import time
from types import SimpleNamespace

import numpy as np
import torch
from scipy.io import mmread
from scipy.sparse.linalg import LinearOperator, bicgstab, spilu


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model_neuro_fsai import Net
from pcg import build_neuro_fsai_csr
from utils.convert_suitesparse import canonicalize_sparse_matrix, matrix_to_graph


SCALES = [
    ("s1", 100, 499),
    ("s2", 500, 999),
    ("s3", 1000, 1999),
    ("s4", 2000, 4999),
    ("s5", 5000, 20000),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selected-json", default="results/selected_eval_matrices_hdrive_train_excluded.json")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--spilu-drop-tol", type=float, default=1e-4)
    parser.add_argument("--spilu-fill-factor", type=float, default=10.0)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--hidden-layers-encoder", type=int, default=1)
    parser.add_argument("--hidden-layers-processor", type=int, default=1)
    parser.add_argument("--hidden-layers-decoder", type=int, default=1)
    parser.add_argument("--fsai-offdiag-scale", type=float, default=0.1)
    parser.add_argument("--fsai-offdiag-basis-cap", type=float, default=1.0)
    parser.add_argument("--fsai-diag-abs-floor", type=float, default=1e-2)
    parser.add_argument("--fsai-diag-scale", type=float, default=0.0)
    parser.add_argument("--fsai-jacobi-eps", type=float, default=1e-12)
    parser.add_argument("--fsai-relative-value-clip", type=float, default=10.0)
    parser.add_argument(
        "--topology-hop",
        type=int,
        default=1,
        choices=[1, 2],
        help="candidate FSAI topology for model input: 1=A pattern, 2=union of A and A^2 patterns")
    parser.add_argument(
        "--max-topology-edges",
        type=int,
        default=0,
        help="fallback to 1-hop when expanded topology exceeds this edge count; <=0 disables")
    parser.add_argument(
        "--max-topology-ratio",
        type=float,
        default=0.0,
        help="fallback to 1-hop when expanded/original edge ratio exceeds this value; <=0 disables")
    parser.add_argument("--output", default="results/suitesparse_eval_fsai_report.json")
    parser.add_argument("--partial-jsonl", default=None)
    return parser.parse_args()


def scale_of(n):
    for label, lo, hi in SCALES:
        if lo <= n <= hi:
            return label
    return "other"


def read_matrix_from_tar(tar_path):
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".mtx"):
                fh = tar.extractfile(member)
                if fh is not None:
                    return mmread(io.BytesIO(fh.read()))
    raise FileNotFoundError(f"no .mtx found in {tar_path}")


def build_model(args):
    model_args = SimpleNamespace(
        dataset="suitesparse",
        hidden_dim=args.hidden_dim,
        hidden_layers_encoder=args.hidden_layers_encoder,
        hidden_layers_decoder=args.hidden_layers_decoder,
        hidden_layers_processor=args.hidden_layers_processor,
        num_iterations=args.num_iterations,
        norm="LayerNorm",
        use_global=False,
        use_r=False,
        diagonalize=False,
        use_pred_x=True,
        fsai_offdiag_scale=args.fsai_offdiag_scale,
        fsai_offdiag_basis_cap=args.fsai_offdiag_basis_cap,
        fsai_diag_abs_floor=args.fsai_diag_abs_floor,
        fsai_diag_scale=args.fsai_diag_scale,
        fsai_jacobi_eps=args.fsai_jacobi_eps,
        fsai_relative_value_clip=args.fsai_relative_value_clip,
    )
    return Net(
        model_args,
        in_dim_node=4,
        in_dim_edge=3,
        out_dim=1,
        b_dim=1,
        num_edges=1,
        out_dim_node=args.hidden_dim,
        out_dim_edge=args.hidden_dim,
        hidden_dim_node=args.hidden_dim,
        hidden_dim_edge=args.hidden_dim,
        hidden_layers_node=args.hidden_layers_encoder,
        hidden_layers_edge=args.hidden_layers_encoder,
        num_iterations=args.num_iterations,
        hidden_dim_processor_node=args.hidden_dim,
        hidden_dim_processor_edge=args.hidden_dim,
        hidden_layers_processor_node=args.hidden_layers_processor,
        hidden_layers_processor_edge=args.hidden_layers_processor,
        hidden_dim_decoder=args.hidden_dim,
        hidden_layers_decoder=args.hidden_layers_decoder,
        dirichlet_idx=3,
        norm_type="LayerNorm",
    )


def run_bicgstab(A_csr, b_np, matvec, max_iter, rtol):
    counter = {"iters": 0}

    def callback(_xk):
        counter["iters"] += 1

    M = LinearOperator(A_csr.shape, matvec=matvec, dtype=np.float64)
    start = time.perf_counter()
    _x, info = bicgstab(
        A_csr, b_np, x0=np.zeros(A_csr.shape[0]),
        rtol=rtol, atol=0.0, maxiter=max_iter,
        M=M, callback=callback)
    solve_time = time.perf_counter() - start
    return (counter["iters"] if info == 0 else max_iter), solve_time, int(info)


def failure_result(max_iter, setup_time=0.0, error=None):
    return {
        "iterations": max_iter,
        "setup_time": float(setup_time),
        "solve_time": 0.0,
        "total_time": float(setup_time),
        "info": -999,
        "error": error,
    }


def log_result(log, method, result):
    error = result.get("error")
    suffix = f", error={error}" if error else ""
    log(
        f"  {method}: info={result['info']} iter={result['iterations']} "
        f"setup={result['setup_time']:.6f}s solve={result['solve_time']:.6f}s "
        f"total={result['total_time']:.6f}s{suffix}"
    )


def evaluate_matrix(model, tar_path, device, max_iter, rtol, spilu_drop_tol, spilu_fill_factor,
                    topology_hop, max_topology_edges, max_topology_ratio, log=print):
    matrix_start = time.perf_counter()
    log("  read_matrix: start")
    matrix = canonicalize_sparse_matrix(read_matrix_from_tar(tar_path))
    log(f"  read_matrix: done {time.perf_counter() - matrix_start:.6f}s")

    graph_start = time.perf_counter()
    graph = matrix_to_graph(
        matrix, name=pathlib.Path(tar_path).stem.replace(".tar", ""),
        topology_hop=topology_hop,
        max_topology_edges=max_topology_edges,
        max_topology_ratio=max_topology_ratio)
    x = torch.as_tensor(graph["x"]).float().to(device)
    x[:, 3] = 0.0

    edge_attr_raw = torch.as_tensor(graph["edge_attr"]).float()
    edge_attr = torch.stack([edge_attr_raw[:, 0], edge_attr_raw[:, 2], edge_attr_raw[:, 1]], dim=-1)
    edge_vals = edge_attr[:, -1]
    scale = max(1.0, edge_vals.abs().max().item())
    edge_attr[:, -1] = edge_attr[:, -1] / scale
    edge_attr = edge_attr.to(device)

    edge_index = torch.as_tensor(graph["edge_index"]).long().to(device)
    rhs = (torch.as_tensor(graph["rhs"]).float() / scale).to(device)
    diag = (torch.as_tensor(graph["diag"]).reshape(-1, 1).float() / scale).to(device)
    r = torch.as_tensor(graph["r"]).float().to(device)
    u_next = torch.as_tensor(graph["u_next"]).float().to(device)
    batch_idx = torch.zeros(x.shape[0], dtype=torch.long, device=device)
    graph_prep_time = time.perf_counter() - graph_start
    log(f"  graph_prep: done {graph_prep_time:.6f}s")

    with torch.no_grad():
        log("  fsai_inference: start")
        inference_start = time.perf_counter()
        _, _, (G_L_ei, G_L_val), (G_U_ei, G_U_val), _ = model(
            x, edge_attr, edge_index,
            diag=diag,
            input_r=r,
            input_x=torch.zeros_like(u_next),
            batch_idx=batch_idx,
            include_r=False,
            use_global=False,
            diagonalize=False,
            use_pred_x=True,
        )
        fsai_inference_time = time.perf_counter() - inference_start
        log(f"  fsai_inference: done {fsai_inference_time:.6f}s")

    N = x.shape[0]
    A_csr = (matrix / scale).astype(np.float64).tocsr()
    b_np = rhs.detach().cpu().numpy().ravel().astype(np.float64)

    factor_build_start = time.perf_counter()
    G_L_csr, G_U_csr = build_neuro_fsai_csr(
        G_L_ei, G_L_val, G_U_ei, G_U_val, N,
        dirichlet_mask=torch.zeros(N, dtype=torch.bool))
    factor_build_time = time.perf_counter() - factor_build_start
    fsai_setup = graph_prep_time + fsai_inference_time + factor_build_time
    log(f"  fsai_factor_build: done {factor_build_time:.6f}s")

    def fsai_apply(vec):
        return G_U_csr @ (G_L_csr @ vec)

    try:
        log("  neuro_fsai: solve start")
        fsai_iter, fsai_solve, fsai_info = run_bicgstab(A_csr, b_np, fsai_apply, max_iter, rtol)
        fsai_result = {
            "iterations": fsai_iter,
            "setup_time": fsai_setup,
            "solve_time": fsai_solve,
            "total_time": fsai_setup + fsai_solve,
            "info": fsai_info,
            "graph_prep_time": graph_prep_time,
            "inference_time": fsai_inference_time,
            "factor_build_time": factor_build_time,
        }
    except Exception as exc:
        fsai_result = failure_result(max_iter, setup_time=fsai_setup, error=str(exc))
        fsai_result["graph_prep_time"] = graph_prep_time
        fsai_result["inference_time"] = fsai_inference_time
        fsai_result["factor_build_time"] = factor_build_time
    log_result(log, "neuro_fsai", fsai_result)

    ilu_setup_start = time.perf_counter()
    try:
        log("  ilu0: setup start")
        ilu = spilu(A_csr.tocsc(), drop_tol=0.0, fill_factor=1.0)
        ilu_setup = time.perf_counter() - ilu_setup_start
        log(f"  ilu0: solve start after setup={ilu_setup:.6f}s")
        ilu_iter, ilu_solve, ilu_info = run_bicgstab(A_csr, b_np, ilu.solve, max_iter, rtol)
        ilu_result = {
            "iterations": ilu_iter,
            "setup_time": ilu_setup,
            "solve_time": ilu_solve,
            "total_time": ilu_setup + ilu_solve,
            "info": ilu_info,
        }
    except Exception as exc:
        ilu_setup = time.perf_counter() - ilu_setup_start
        ilu_result = failure_result(max_iter, setup_time=ilu_setup, error=str(exc))
    log_result(log, "ilu0", ilu_result)

    ilut_setup_start = time.perf_counter()
    try:
        log("  spilu: setup start")
        ilut = spilu(A_csr.tocsc(), drop_tol=spilu_drop_tol, fill_factor=spilu_fill_factor)
        ilut_setup = time.perf_counter() - ilut_setup_start
        log(f"  spilu: solve start after setup={ilut_setup:.6f}s")
        ilut_iter, ilut_solve, ilut_info = run_bicgstab(A_csr, b_np, ilut.solve, max_iter, rtol)
        ilut_result = {
            "iterations": ilut_iter,
            "setup_time": ilut_setup,
            "solve_time": ilut_solve,
            "total_time": ilut_setup + ilut_solve,
            "info": ilut_info,
        }
    except Exception as exc:
        ilut_setup = time.perf_counter() - ilut_setup_start
        ilut_result = failure_result(max_iter, setup_time=ilut_setup, error=str(exc))
    log_result(log, "spilu", ilut_result)

    diag_vals = A_csr.diagonal().astype(np.float64)
    diag_vals[diag_vals == 0.0] = 1.0
    try:
        log("  jacobi: solve start")
        jacobi_iter, jacobi_solve, jacobi_info = run_bicgstab(
            A_csr, b_np, lambda v: v / diag_vals, max_iter, rtol)
        jacobi_result = {
            "iterations": jacobi_iter,
            "setup_time": 0.0,
            "solve_time": jacobi_solve,
            "total_time": jacobi_solve,
            "info": jacobi_info,
        }
    except Exception as exc:
        jacobi_result = failure_result(max_iter, setup_time=0.0, error=str(exc))
    log_result(log, "jacobi", jacobi_result)

    try:
        log("  identity: solve start")
        identity_iter, identity_solve, identity_info = run_bicgstab(
            A_csr, b_np, lambda v: v, max_iter, rtol)
        identity_result = {
            "iterations": identity_iter,
            "setup_time": 0.0,
            "solve_time": identity_solve,
            "total_time": identity_solve,
            "info": identity_info,
        }
    except Exception as exc:
        identity_result = failure_result(max_iter, setup_time=0.0, error=str(exc))
    log_result(log, "identity", identity_result)

    return {
        "name": graph["meta"]["name"],
        "nrows": int(matrix.shape[0]),
        "nnz": int(matrix.nnz),
        "scale": scale_of(int(matrix.shape[0])),
        "neuro_fsai": fsai_result,
        "ilu0": ilu_result,
        "spilu": ilut_result,
        "jacobi": jacobi_result,
        "identity": identity_result,
    }


def summarize(rows):
    summary = {"overall": {}, "by_scale": {}}
    valid_rows = [r for r in rows if "neuro_fsai" in r]
    for scope_name, scoped_rows in {
        "overall": valid_rows,
        **{label: [r for r in valid_rows if r["scale"] == label] for label, _, _ in SCALES},
    }.items():
        if not scoped_rows:
            continue
        block = {}
        for method in ["neuro_fsai", "ilu0", "spilu", "jacobi", "identity"]:
            success_mask = [r[method]["info"] == 0 for r in scoped_rows]
            block[method] = {
                "avg_iterations": float(np.mean([r[method]["iterations"] for r in scoped_rows])),
                "avg_setup_time": float(np.mean([r[method]["setup_time"] for r in scoped_rows])),
                "avg_solve_time": float(np.mean([r[method]["solve_time"] for r in scoped_rows])),
                "avg_total_time": float(np.mean([r[method]["total_time"] for r in scoped_rows])),
                "median_iterations": float(np.median([r[method]["iterations"] for r in scoped_rows])),
                "success_count": int(sum(success_mask)),
                "failure_count": int(len(success_mask) - sum(success_mask)),
            }
        if scope_name == "overall":
            summary["overall"] = block
        else:
            summary["by_scale"][scope_name] = block
    return summary


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log = lambda message: print(message, flush=True)
    log(f"Loading checkpoint: {args.checkpoint}")
    log(f"Device: {device}")
    model = build_model(args).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    with open(args.selected_json) as f:
        selected = json.load(f)["scales"]

    rows = []
    partial_path = pathlib.Path(args.partial_jsonl) if args.partial_jsonl else pathlib.Path(args.output).with_suffix(".partial.jsonl")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.write_text("")
    for scale_rows in selected.values():
        for record in scale_rows:
            log(f"Evaluating {record['name']} ({record['nrows']}x{record['ncols']}, nnz={record['nnz']})")
            try:
                row = evaluate_matrix(
                    model, record["copied_to"], device,
                    args.max_iter, args.rtol,
                    args.spilu_drop_tol, args.spilu_fill_factor,
                    args.topology_hop,
                    args.max_topology_edges,
                    args.max_topology_ratio,
                    log=log)
            except Exception as exc:
                row = {
                    "name": record["name"],
                    "nrows": int(record["nrows"]),
                    "nnz": int(record["nnz"]),
                    "scale": scale_of(int(record["nrows"])),
                    "error": str(exc),
                }
                log(f"  failed: {exc}")
            rows.append(row)
            with partial_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            log(f"Finished {record['name']}")

    report = {
        "checkpoint": args.checkpoint,
        "selected_json": args.selected_json,
        "device": str(device),
        "max_iter": args.max_iter,
        "rtol": args.rtol,
        "spilu_drop_tol": args.spilu_drop_tol,
        "spilu_fill_factor": args.spilu_fill_factor,
        "topology_hop": args.topology_hop,
        "max_topology_edges": args.max_topology_edges,
        "max_topology_ratio": args.max_topology_ratio,
        "matrix_count": len(rows),
        "results": rows,
        "summary": summarize(rows),
    }

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    log(json.dumps(report["summary"], indent=2))
    log(f"\nSaved report to {output_path}")
    log(f"Saved partial rows to {partial_path}")


if __name__ == "__main__":
    main()
