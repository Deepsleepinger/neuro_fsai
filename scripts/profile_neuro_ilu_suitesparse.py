import argparse
import io
import json
import pathlib
import sys
import tarfile
import time

import numpy as np
import torch
from scipy.io import mmread
from scipy.sparse.linalg import LinearOperator, bicgstab, spsolve_triangular


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcg import build_neuro_ilu_csr
from scripts.evaluate_neuro_ilu_suitesparse import build_model, scale_of
from utils.convert_suitesparse import canonicalize_sparse_matrix, matrix_to_graph


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/mnt/h/neuro_ilu/checkpoints/suitesparse_run3-neuroilu-20260611-164503/model/best_val.pt")
    parser.add_argument("--selected-json", default="results/selected_eval_matrices.json")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--output", default="results/neuro_ilu_profile_report.json")
    return parser.parse_args()


def read_matrix_from_tar(tar_path):
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".mtx"):
                fh = tar.extractfile(member)
                if fh is not None:
                    return mmread(io.BytesIO(fh.read()))
    raise FileNotFoundError(f"no .mtx found in {tar_path}")


def prepare_graph(tar_path, device):
    matrix = canonicalize_sparse_matrix(read_matrix_from_tar(tar_path))
    graph = matrix_to_graph(matrix, name=pathlib.Path(tar_path).stem.replace(".tar", ""))

    x = torch.as_tensor(graph["x"]).float().to(device)
    x[:, 3] = 0.0

    edge_attr_raw = torch.as_tensor(graph["edge_attr"]).float()
    edge_attr = torch.stack([edge_attr_raw[:, 0], edge_attr_raw[:, 2], edge_attr_raw[:, 1]], dim=-1)
    scale = max(1.0, edge_attr[:, -1].abs().max().item())
    edge_attr[:, -1] = edge_attr[:, -1] / scale

    edge_index = torch.as_tensor(graph["edge_index"]).long().to(device)
    rhs = (torch.as_tensor(graph["rhs"]).float() / scale).to(device)
    diag = (torch.as_tensor(graph["diag"]).reshape(-1, 1).float() / scale).to(device)
    r = torch.as_tensor(graph["r"]).float().to(device)
    u_next = torch.as_tensor(graph["u_next"]).float().to(device)
    batch_idx = torch.zeros(x.shape[0], dtype=torch.long, device=device)

    return {
        "matrix": matrix,
        "graph": graph,
        "x": x,
        "edge_attr": edge_attr.to(device),
        "edge_index": edge_index,
        "rhs": rhs,
        "diag": diag,
        "r": r,
        "u_next": u_next,
        "batch_idx": batch_idx,
        "scale": scale,
    }


def profile_matrix(model, tar_path, device, max_iter, rtol):
    prepared = prepare_graph(tar_path, device)
    matrix = prepared["matrix"]
    graph = prepared["graph"]
    scale = prepared["scale"]

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_start = time.perf_counter()
    with torch.no_grad():
        _, _, (L_ei, L_val), (U_ei, U_val), _ = model(
            prepared["x"], prepared["edge_attr"], prepared["edge_index"],
            diag=prepared["diag"],
            input_r=prepared["r"],
            input_x=torch.zeros_like(prepared["u_next"]),
            batch_idx=prepared["batch_idx"],
            include_r=False,
            use_global=False,
            diagonalize=False,
            use_pred_x=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_time = time.perf_counter() - forward_start

    N = prepared["x"].shape[0]
    csr_start = time.perf_counter()
    L_csr, U_csr = build_neuro_ilu_csr(
        L_ei, L_val, U_ei, U_val, N,
        dirichlet_mask=torch.zeros(N, dtype=torch.bool))
    A_csr = (matrix / scale).astype(np.float64).tocsr()
    b_np = prepared["rhs"].detach().cpu().numpy().ravel().astype(np.float64)
    csr_build_time = time.perf_counter() - csr_start

    stats = {
        "preconditioner_calls": 0,
        "preconditioner_total_time": 0.0,
        "lower_solve_time": 0.0,
        "upper_solve_time": 0.0,
    }

    def neuro_apply(vec):
        stats["preconditioner_calls"] += 1

        start = time.perf_counter()
        w = spsolve_triangular(L_csr, vec, lower=True, unit_diagonal=True)
        lower_time = time.perf_counter() - start

        start = time.perf_counter()
        out = spsolve_triangular(U_csr, w, lower=False, unit_diagonal=False)
        upper_time = time.perf_counter() - start

        stats["lower_solve_time"] += lower_time
        stats["upper_solve_time"] += upper_time
        stats["preconditioner_total_time"] += lower_time + upper_time
        return out

    counter = {"iters": 0}

    def callback(_xk):
        counter["iters"] += 1

    operator = LinearOperator(A_csr.shape, matvec=neuro_apply, dtype=np.float64)

    solve_start = time.perf_counter()
    _x, info = bicgstab(
        A_csr, b_np,
        x0=np.zeros(A_csr.shape[0]),
        rtol=rtol,
        atol=0.0,
        maxiter=max_iter,
        M=operator,
        callback=callback,
    )
    solve_time = time.perf_counter() - solve_start

    iterations = counter["iters"] if info == 0 else max_iter
    stats["other_solve_time"] = solve_time - stats["preconditioner_total_time"]
    stats["avg_preconditioner_time"] = (
        stats["preconditioner_total_time"] / stats["preconditioner_calls"]
        if stats["preconditioner_calls"] > 0 else 0.0
    )
    stats["avg_lower_solve_time"] = (
        stats["lower_solve_time"] / stats["preconditioner_calls"]
        if stats["preconditioner_calls"] > 0 else 0.0
    )
    stats["avg_upper_solve_time"] = (
        stats["upper_solve_time"] / stats["preconditioner_calls"]
        if stats["preconditioner_calls"] > 0 else 0.0
    )
    stats["avg_iteration_wall_time"] = solve_time / max(1, iterations)

    return {
        "name": graph["meta"]["name"],
        "scale": scale_of(int(matrix.shape[0])),
        "nrows": int(matrix.shape[0]),
        "nnz": int(matrix.nnz),
        "info": int(info),
        "iterations": int(iterations),
        "forward_time": forward_time,
        "csr_build_time": csr_build_time,
        "solve_time": solve_time,
        "total_time": forward_time + csr_build_time + solve_time,
        **stats,
    }


def summarize(rows):
    if not rows:
        return {}

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    return {
        "matrix_count": len(rows),
        "success_count": int(sum(r["info"] == 0 for r in rows)),
        "failure_count": int(sum(r["info"] != 0 for r in rows)),
        "avg_iterations": mean("iterations"),
        "avg_forward_time": mean("forward_time"),
        "avg_csr_build_time": mean("csr_build_time"),
        "avg_solve_time": mean("solve_time"),
        "avg_total_time": mean("total_time"),
        "avg_preconditioner_calls": mean("preconditioner_calls"),
        "avg_preconditioner_total_time": mean("preconditioner_total_time"),
        "avg_other_solve_time": mean("other_solve_time"),
        "avg_preconditioner_time": mean("avg_preconditioner_time"),
        "avg_lower_solve_time": mean("avg_lower_solve_time"),
        "avg_upper_solve_time": mean("avg_upper_solve_time"),
        "avg_iteration_wall_time": mean("avg_iteration_wall_time"),
    }


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    selected = json.loads(pathlib.Path(args.selected_json).read_text())["scales"]

    first_record = next(iter(next(iter(selected.values()))), None)
    if first_record is not None:
        print(f"Warming up on {first_record['name']}")
        _ = profile_matrix(model, first_record["copied_to"], device, max_iter=2, rtol=args.rtol)

    rows = []
    for scale_rows in selected.values():
        for record in scale_rows:
            print(f"Profiling {record['name']} ({record['nrows']}x{record['ncols']}, nnz={record['nnz']})")
            rows.append(profile_matrix(model, record["copied_to"], device, args.max_iter, args.rtol))

    report = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "results": rows,
        "summary": summarize(rows),
    }

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
