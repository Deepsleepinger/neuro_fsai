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
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import LinearOperator, bicgstab, spilu, spsolve_triangular


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model_neuro_ilu import Net
from pcg import build_neuro_ilu_csr
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
    parser.add_argument("--checkpoint", default="/mnt/h/neuro_ilu/checkpoints/suitesparse_run3-neuroilu-20260611-164503/model/best_val.pt")
    parser.add_argument("--selected-json", default="results/selected_eval_matrices.json")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--spilu-drop-tol", type=float, default=1e-4)
    parser.add_argument("--spilu-fill-factor", type=float, default=10.0)
    parser.add_argument("--output", default="results/suitesparse_eval_report.json")
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


def build_model():
    args = SimpleNamespace(
        dataset="suitesparse",
        hidden_dim=16,
        hidden_layers_encoder=1,
        hidden_layers_decoder=1,
        hidden_layers_processor=1,
        num_iterations=5,
        norm="LayerNorm",
        use_global=False,
        use_r=False,
        diagonalize=False,
        use_pred_x=True,
    )
    model = Net(
        args,
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
        norm_type=args.norm,
    )
    return model


def run_bicgstab(A_csr, b_np, matvec, max_iter, rtol):
    counter = {"iters": 0}

    def callback(_xk):
        counter["iters"] += 1

    M = LinearOperator(A_csr.shape, matvec=matvec, dtype=np.float64)
    start = time.perf_counter()
    _x, info = bicgstab(A_csr, b_np, x0=np.zeros(A_csr.shape[0]), rtol=rtol, atol=0.0, maxiter=max_iter, M=M, callback=callback)
    solve_time = time.perf_counter() - start
    return (counter["iters"] if info == 0 else max_iter), solve_time, int(info)


def stabilize_u_diagonal(U_csr, A_csr, floor_rel, floor_abs):
    """Clamp tiny U diagonal entries relative to the original A diagonal."""
    u_diag = U_csr.diagonal().astype(np.float64)
    a_diag = A_csr.diagonal().astype(np.float64)

    target = np.maximum(floor_rel * np.abs(a_diag), floor_abs)
    sign = np.sign(u_diag)
    zero_mask = sign == 0.0
    sign[zero_mask] = np.sign(a_diag[zero_mask])
    sign[sign == 0.0] = 1.0

    clipped_mask = np.abs(u_diag) < target
    if not clipped_mask.any():
        return U_csr, 0, float(np.abs(u_diag).min()) if u_diag.size else 0.0, float(np.abs(u_diag).min()) if u_diag.size else 0.0

    new_diag = u_diag.copy()
    new_diag[clipped_mask] = sign[clipped_mask] * target[clipped_mask]

    U_mod = U_csr.copy()
    U_mod.setdiag(new_diag)
    U_mod.eliminate_zeros()
    return U_mod.tocsr(), int(clipped_mask.sum()), float(np.abs(u_diag).min()), float(np.abs(new_diag).min())


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


def evaluate_matrix(model, tar_path, device, max_iter, rtol, spilu_drop_tol, spilu_fill_factor, log=print):
    matrix_start = time.perf_counter()
    log("  read_matrix: start")
    matrix = read_matrix_from_tar(tar_path)
    matrix = canonicalize_sparse_matrix(matrix)
    log(f"  read_matrix: done {time.perf_counter() - matrix_start:.6f}s")

    graph_start = time.perf_counter()
    graph = matrix_to_graph(matrix, name=pathlib.Path(tar_path).stem.replace(".tar", ""))
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
        log("  neuro_inference: start")
        inference_start = time.perf_counter()
        _, _, (L_ei, L_val), (U_ei, U_val), _ = model(
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
        neuro_inference_time = time.perf_counter() - inference_start
        log(f"  neuro_inference: done {neuro_inference_time:.6f}s")

    N = x.shape[0]
    A_csr = (matrix / scale).astype(np.float64).tocsr()
    b_np = rhs.detach().cpu().numpy().ravel().astype(np.float64)

    factor_build_start = time.perf_counter()
    L_csr, U_csr = build_neuro_ilu_csr(L_ei, L_val, U_ei, U_val, N, dirichlet_mask=torch.zeros(N, dtype=torch.bool))
    factor_build_time = time.perf_counter() - factor_build_start
    neuro_setup = graph_prep_time + neuro_inference_time + factor_build_time
    log(f"  neuro_factor_build: done {factor_build_time:.6f}s")

    diagfix_start = time.perf_counter()
    U_diagfix_csr, clipped_diag_count, u_diag_min_before, u_diag_min_after = stabilize_u_diagonal(
        U_csr, A_csr, floor_rel=0.1, floor_abs=1e-3)
    diagfix_setup_time = time.perf_counter() - diagfix_start
    neuro_diagfix_setup = neuro_setup + diagfix_setup_time

    def neuro_apply(vec):
        w = spsolve_triangular(L_csr, vec, lower=True, unit_diagonal=True)
        return spsolve_triangular(U_csr, w, lower=False, unit_diagonal=False)

    def neuro_diagfix_apply(vec):
        w = spsolve_triangular(L_csr, vec, lower=True, unit_diagonal=True)
        return spsolve_triangular(U_diagfix_csr, w, lower=False, unit_diagonal=False)

    try:
        log("  neuro: solve start")
        neuro_iter, neuro_solve, neuro_info = run_bicgstab(A_csr, b_np, neuro_apply, max_iter, rtol)
        neuro_result = {
            "iterations": neuro_iter,
            "setup_time": neuro_setup,
            "solve_time": neuro_solve,
            "total_time": neuro_setup + neuro_solve,
            "info": neuro_info,
            "graph_prep_time": graph_prep_time,
            "inference_time": neuro_inference_time,
            "factor_build_time": factor_build_time,
        }
    except Exception as exc:
        neuro_result = failure_result(max_iter, setup_time=neuro_setup, error=str(exc))
        neuro_result["graph_prep_time"] = graph_prep_time
        neuro_result["inference_time"] = neuro_inference_time
        neuro_result["factor_build_time"] = factor_build_time
    log_result(log, "neuro", neuro_result)

    try:
        log("  neuro_diagfix: solve start")
        neuro_diagfix_iter, neuro_diagfix_solve, neuro_diagfix_info = run_bicgstab(
            A_csr, b_np, neuro_diagfix_apply, max_iter, rtol)
        neuro_diagfix_result = {
            "iterations": neuro_diagfix_iter,
            "setup_time": neuro_diagfix_setup,
            "solve_time": neuro_diagfix_solve,
            "total_time": neuro_diagfix_setup + neuro_diagfix_solve,
            "info": neuro_diagfix_info,
            "clipped_diag_count": clipped_diag_count,
            "u_diag_abs_min_before": u_diag_min_before,
            "u_diag_abs_min_after": u_diag_min_after,
            "graph_prep_time": graph_prep_time,
            "inference_time": neuro_inference_time,
            "factor_build_time": factor_build_time,
            "diagfix_setup_time": diagfix_setup_time,
        }
    except Exception as exc:
        neuro_diagfix_result = failure_result(max_iter, setup_time=neuro_diagfix_setup, error=str(exc))
        neuro_diagfix_result["clipped_diag_count"] = clipped_diag_count
        neuro_diagfix_result["u_diag_abs_min_before"] = u_diag_min_before
        neuro_diagfix_result["u_diag_abs_min_after"] = u_diag_min_after
        neuro_diagfix_result["graph_prep_time"] = graph_prep_time
        neuro_diagfix_result["inference_time"] = neuro_inference_time
        neuro_diagfix_result["factor_build_time"] = factor_build_time
        neuro_diagfix_result["diagfix_setup_time"] = diagfix_setup_time
    log_result(log, "neuro_diagfix", neuro_diagfix_result)

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
    jacobi_setup = 0.0
    try:
        log("  jacobi: solve start")
        jacobi_iter, jacobi_solve, jacobi_info = run_bicgstab(A_csr, b_np, lambda v: v / diag_vals, max_iter, rtol)
        jacobi_result = {
            "iterations": jacobi_iter,
            "setup_time": jacobi_setup,
            "solve_time": jacobi_solve,
            "total_time": jacobi_setup + jacobi_solve,
            "info": jacobi_info,
        }
    except Exception as exc:
        jacobi_result = failure_result(max_iter, setup_time=jacobi_setup, error=str(exc))
    log_result(log, "jacobi", jacobi_result)

    identity_setup = 0.0
    try:
        log("  identity: solve start")
        identity_iter, identity_solve, identity_info = run_bicgstab(A_csr, b_np, lambda v: v, max_iter, rtol)
        identity_result = {
            "iterations": identity_iter,
            "setup_time": identity_setup,
            "solve_time": identity_solve,
            "total_time": identity_setup + identity_solve,
            "info": identity_info,
        }
    except Exception as exc:
        identity_result = failure_result(max_iter, setup_time=identity_setup, error=str(exc))
    log_result(log, "identity", identity_result)

    return {
        "name": graph["meta"]["name"],
        "nrows": int(matrix.shape[0]),
        "nnz": int(matrix.nnz),
        "scale": scale_of(int(matrix.shape[0])),
        "neuro": neuro_result,
        "neuro_diagfix": neuro_diagfix_result,
        "ilu0": ilu_result,
        "spilu": ilut_result,
        "jacobi": jacobi_result,
        "identity": identity_result,
    }


def summarize(rows):
    summary = {"overall": {}, "by_scale": {}}
    valid_rows = [r for r in rows if "neuro" in r]
    for scope_name, scoped_rows in {
        "overall": valid_rows,
        **{label: [r for r in valid_rows if r["scale"] == label] for label, _, _ in SCALES},
    }.items():
        if not scoped_rows:
            continue
        block = {}
        for method in ["neuro", "neuro_diagfix", "ilu0", "spilu", "jacobi", "identity"]:
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
    model = build_model().to(device)
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
