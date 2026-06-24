import argparse
import io
import json
import pathlib
import sys
import tarfile

import numpy as np
import torch
from scipy.io import mmread
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve_triangular


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcg import build_neuro_ilu_csr
from scripts.evaluate_neuro_ilu_suitesparse import build_model, scale_of
from utils.convert_suitesparse import canonicalize_sparse_matrix, matrix_to_graph


RNG_SEED = 0
NUM_RANDOM_TESTS = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/mnt/h/neuro_ilu/checkpoints/suitesparse_run3-neuroilu-20260611-164503/model/best_val.pt")
    parser.add_argument("--selected-json", default="results/selected_eval_matrices.json")
    parser.add_argument("--benchmark-report", default="results/suitesparse_eval_report_spilu.json")
    parser.add_argument("--output", default="results/neuro_ilu_factor_diagnostics.json")
    return parser.parse_args()


def read_matrix_from_tar(tar_path):
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".mtx"):
                fh = tar.extractfile(member)
                if fh is not None:
                    return mmread(io.BytesIO(fh.read()))
    raise FileNotFoundError(f"no .mtx found in {tar_path}")


def sparse_fro_norm(A):
    if A.nnz == 0:
        return 0.0
    return float(np.sqrt(np.dot(A.data, A.data)))


def prepare_inputs(tar_path, device):
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
        "matrix": (matrix / scale).astype(np.float64).tocsr(),
        "graph": graph,
        "x": x,
        "edge_attr": edge_attr.to(device),
        "edge_index": edge_index,
        "rhs": rhs.detach().cpu().numpy().ravel().astype(np.float64),
        "diag": diag,
        "r": r,
        "u_next": u_next,
        "u_true": np.asarray(graph["u"], dtype=np.float64).ravel(),
        "batch_idx": batch_idx,
    }


def apply_preconditioner(L_csr, U_csr, vec):
    w = spsolve_triangular(L_csr, vec, lower=True, unit_diagonal=True)
    return spsolve_triangular(U_csr, w, lower=False, unit_diagonal=False)


def compute_random_identity_errors(A_csr, L_csr, U_csr, rng):
    left_errors = []
    right_errors = []
    for _ in range(NUM_RANDOM_TESTS):
        v = rng.standard_normal(A_csr.shape[0]).astype(np.float64)
        v_norm = np.linalg.norm(v)
        if v_norm == 0.0:
            continue

        left_vec = apply_preconditioner(L_csr, U_csr, A_csr @ v)
        left_errors.append(float(np.linalg.norm(left_vec - v) / v_norm))

        right_vec = A_csr @ apply_preconditioner(L_csr, U_csr, v)
        right_errors.append(float(np.linalg.norm(right_vec - v) / v_norm))

    return {
        "left_identity_error_mean": float(np.mean(left_errors)) if left_errors else 0.0,
        "left_identity_error_max": float(np.max(left_errors)) if left_errors else 0.0,
        "right_identity_error_mean": float(np.mean(right_errors)) if right_errors else 0.0,
        "right_identity_error_max": float(np.max(right_errors)) if right_errors else 0.0,
    }


def diagnose_matrix(model, tar_path, device, benchmark_row=None):
    prepared = prepare_inputs(tar_path, device)

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

    N = prepared["x"].shape[0]
    A_csr = prepared["matrix"]
    A_norm = max(sparse_fro_norm(A_csr), 1e-16)

    L_csr, U_csr = build_neuro_ilu_csr(
        L_ei, L_val, U_ei, U_val, N,
        dirichlet_mask=torch.zeros(N, dtype=torch.bool))

    M_csr = (L_csr @ U_csr).tocsr()
    M_csr.sum_duplicates()
    M_csr.eliminate_zeros()

    diff = (M_csr - A_csr).tocsr()
    diff.sum_duplicates()
    diff.eliminate_zeros()

    A_mask = A_csr.copy()
    A_mask.data = np.ones_like(A_mask.data)
    M_on_pattern = M_csr.multiply(A_mask).tocsr()
    fill_only = (M_csr - M_on_pattern).tocsr()
    pattern_residual = (M_on_pattern - A_csr).tocsr()

    U_diag = U_csr.diagonal().astype(np.float64)
    abs_U_diag = np.abs(U_diag)

    x0 = apply_preconditioner(L_csr, U_csr, prepared["rhs"])
    rhs_norm = max(np.linalg.norm(prepared["rhs"]), 1e-16)
    u_norm = max(np.linalg.norm(prepared["u_true"]), 1e-16)
    one_step_residual = A_csr @ x0 - prepared["rhs"]

    rng = np.random.default_rng(RNG_SEED)
    identity_errors = compute_random_identity_errors(A_csr, L_csr, U_csr, rng)

    row = {
        "name": prepared["graph"]["meta"]["name"],
        "scale": scale_of(int(A_csr.shape[0])),
        "nrows": int(A_csr.shape[0]),
        "nnz_A": int(A_csr.nnz),
        "nnz_L": int(L_csr.nnz),
        "nnz_U": int(U_csr.nnz),
        "nnz_LU": int(M_csr.nnz),
        "lu_fill_ratio_vs_A": float(M_csr.nnz / max(A_csr.nnz, 1)),
        "rel_fro_error_full": sparse_fro_norm(diff) / A_norm,
        "rel_fro_error_on_A_pattern": sparse_fro_norm(pattern_residual) / A_norm,
        "rel_fro_norm_fill_only": sparse_fro_norm(fill_only) / A_norm,
        "u_diag_abs_min": float(abs_U_diag.min()) if abs_U_diag.size else 0.0,
        "u_diag_abs_median": float(np.median(abs_U_diag)) if abs_U_diag.size else 0.0,
        "u_diag_abs_max": float(abs_U_diag.max()) if abs_U_diag.size else 0.0,
        "u_diag_tiny_count_1e-08": int(np.sum(abs_U_diag < 1e-8)),
        "u_diag_tiny_count_1e-06": int(np.sum(abs_U_diag < 1e-6)),
        "u_diag_tiny_count_1e-04": int(np.sum(abs_U_diag < 1e-4)),
        "l_abs_max": float(np.max(np.abs(L_csr.data))) if L_csr.nnz else 0.0,
        "u_abs_max": float(np.max(np.abs(U_csr.data))) if U_csr.nnz else 0.0,
        "rhs_one_step_rel_residual": float(np.linalg.norm(one_step_residual) / rhs_norm),
        "rhs_one_step_rel_solution_error": float(np.linalg.norm(x0 - prepared["u_true"]) / u_norm),
        "rhs_solution_norm_growth": float(np.linalg.norm(x0) / rhs_norm),
        **identity_errors,
    }

    if benchmark_row is not None:
        row["neuro_iterations"] = int(benchmark_row["neuro"]["iterations"])
        row["neuro_info"] = int(benchmark_row["neuro"]["info"])
        row["neuro_total_time"] = float(benchmark_row["neuro"]["total_time"])
        row["spilu_iterations"] = int(benchmark_row["spilu"]["iterations"])
        row["spilu_info"] = int(benchmark_row["spilu"]["info"])
        row["spilu_total_time"] = float(benchmark_row["spilu"]["total_time"])

    return row


def build_summary(rows):
    def subset_stats(name, subset):
        if not subset:
            return {}

        def mean(key):
            return float(np.mean([r[key] for r in subset]))

        return {
            "count": len(subset),
            "avg_rel_fro_error_full": mean("rel_fro_error_full"),
            "avg_rel_fro_error_on_A_pattern": mean("rel_fro_error_on_A_pattern"),
            "avg_rel_fro_norm_fill_only": mean("rel_fro_norm_fill_only"),
            "avg_u_diag_abs_min": mean("u_diag_abs_min"),
            "avg_u_diag_tiny_count_1e-04": mean("u_diag_tiny_count_1e-04"),
            "avg_rhs_one_step_rel_residual": mean("rhs_one_step_rel_residual"),
            "avg_rhs_one_step_rel_solution_error": mean("rhs_one_step_rel_solution_error"),
            "avg_left_identity_error_mean": mean("left_identity_error_mean"),
            "avg_right_identity_error_mean": mean("right_identity_error_mean"),
            "avg_lu_fill_ratio_vs_A": mean("lu_fill_ratio_vs_A"),
        }

    rows_with_conv = [r for r in rows if "neuro_info" in r]
    success_rows = [r for r in rows_with_conv if r["neuro_info"] == 0]
    failure_rows = [r for r in rows_with_conv if r["neuro_info"] != 0]

    return {
        "overall": subset_stats("overall", rows),
        "neuro_success": subset_stats("neuro_success", success_rows),
        "neuro_failure": subset_stats("neuro_failure", failure_rows),
        "worst_by_right_identity_error": [
            {"name": r["name"], "value": r["right_identity_error_mean"]}
            for r in sorted(rows, key=lambda x: x["right_identity_error_mean"], reverse=True)[:5]
        ],
        "worst_by_rhs_one_step_rel_residual": [
            {"name": r["name"], "value": r["rhs_one_step_rel_residual"]}
            for r in sorted(rows, key=lambda x: x["rhs_one_step_rel_residual"], reverse=True)[:5]
        ],
        "worst_by_rel_fro_error_full": [
            {"name": r["name"], "value": r["rel_fro_error_full"]}
            for r in sorted(rows, key=lambda x: x["rel_fro_error_full"], reverse=True)[:5]
        ],
    }


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    benchmark_rows = {}
    benchmark_path = pathlib.Path(args.benchmark_report)
    if benchmark_path.exists():
        benchmark = json.loads(benchmark_path.read_text())
        benchmark_rows = {row["name"]: row for row in benchmark["results"]}

    selected = json.loads(pathlib.Path(args.selected_json).read_text())["scales"]

    rows = []
    for scale_rows in selected.values():
        for record in scale_rows:
            print(f"Diagnosing {record['name']} ({record['nrows']}x{record['ncols']}, nnz={record['nnz']})")
            rows.append(diagnose_matrix(
                model,
                record["copied_to"],
                device,
                benchmark_row=benchmark_rows.get(record["name"]),
            ))

    report = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "results": rows,
        "summary": build_summary(rows),
    }

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
