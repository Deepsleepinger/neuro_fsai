import argparse
import csv
import json
import pathlib
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize Neuro-SPAI experiment metadata into evaluation-table CSVs.")
    parser.add_argument("--run-dirs", nargs="*", default=[])
    parser.add_argument("--end2end-csvs", nargs="*", default=[])
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Defaults to results/evaluation_summary/summary_TIMESTAMP.csv.")
    return parser.parse_args()


def matrix_name(path):
    if not path:
        return None
    return pathlib.Path(path).name.replace(".tar.gz", "")


def infer_mode(run_name, meta, matrix=None):
    name = run_name.lower()
    matrix_text = (matrix or "").lower()
    args = meta.get("args", {})
    if "fewshot" in name or "finetune" in name:
        return "few_shot_online_adaptation"
    if "circuit" in name or "bomhof" in name or "hamm" in name:
        return "same_domain_zero_shot"
    if "rajat27" in name and "multigraph" not in name and "v2" not in name:
        return "capacity_limit_single_matrix"
    if "rajat04" in name or "rajat14" in name or "rajat04" in matrix_text or "rajat14" in matrix_text:
        return "single_matrix_rescue"
    if "v2" in name or "hardneg" in name or args.get("poison_train_replicas", 0):
        return "zero_shot_hard_negative_v2"
    if "multigraph" in name or args.get("eval_matrix_tars"):
        return "zero_shot_transfer"
    return "unspecified"


def infer_tier(matrix, run_name):
    text = f"{matrix} {run_name}".lower()
    if "rajat27" in text:
        return "scale_boss"
    if "rajat" in text:
        return "rescue_mission"
    if "bomhof" in text or "circuit" in text or "hamm" in text:
        return "sanity_circuit"
    if "bai" in text or "ag-monien" in text or "boeing" in text:
        return "hard_negative_auxiliary"
    return "unclassified"


def safe_get_result(row):
    if row is None:
        return {}
    if "result" in row:
        return row.get("result") or {}
    return row


def success(result):
    return bool(result) and int(result.get("info", 1)) == 0


def to_int(value, default=None):
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def csv_success_from_iter(item, prefix, max_iter=2000):
    info = to_int(item.get(f"{prefix}_info"))
    if info is not None:
        return info == 0
    iterations = to_int(item.get(f"{prefix}_iter"))
    if iterations is None:
        return None
    # Older end-to-end CSVs omit solver info; these runs used maxiter=2000.
    return iterations < max_iter


def result_fields(prefix, result):
    result = result or {}
    return {
        f"{prefix}_iter": result.get("iterations"),
        f"{prefix}_info": result.get("info"),
        f"{prefix}_final_rel": result.get("final_rel"),
        f"{prefix}_solve_sec": result.get("solve_time"),
    }


def baseline_for_matrix(meta, matrix):
    if "baselines" in meta:
        return meta.get("baselines") or {}
    for case in meta.get("eval_cases", []) + meta.get("train_cases", []):
        if case.get("name") == matrix:
            return case.get("baselines") or {}
    return {}


def common_config(meta):
    args = meta.get("args", {})
    return {
        "topology_hop": args.get("topology_hop"),
        "topology_row_topk": args.get("topology_row_topk"),
        "amg_levels": args.get("amg_levels"),
        "base_mode": args.get("base_mode"),
        "target_scale_mode": args.get("target_scale_mode"),
        "hutchinson_probe_mode": args.get("hutchinson_probe_mode"),
        "poison_train_replicas": args.get("poison_train_replicas", 0),
        "hidden_dim": args.get("hidden_dim"),
        "num_iterations": args.get("num_iterations"),
    }


def summarize_run_dir(run_dir):
    run_dir = pathlib.Path(run_dir)
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    run_name = run_dir.name
    config = common_config(meta)
    rows = []

    if "best_result" in meta:
        matrix = matrix_name(meta.get("args", {}).get("matrix_tar"))
        mode = infer_mode(run_name, meta, matrix)
        best = safe_get_result(meta.get("best_result"))
        latest = safe_get_result(meta.get("latest_result"))
        baselines = baseline_for_matrix(meta, matrix)
        rows.append(make_row(run_dir, run_name, mode, matrix, meta, best, latest, baselines, config))
        return rows

    best_results = meta.get("best_results") or {}
    latest_results = meta.get("latest_results") or {}
    for matrix, best_row in best_results.items():
        mode = infer_mode(run_name, meta, matrix)
        best = safe_get_result(best_row)
        latest = safe_get_result(latest_results.get(matrix))
        baselines = baseline_for_matrix(meta, matrix)
        rows.append(make_row(run_dir, run_name, mode, matrix, meta, best, latest, baselines, config))
    return rows


def make_row(run_dir, run_name, mode, matrix, meta, best, latest, baselines, config):
    jacobi = baselines.get("jacobi") or {}
    spilu = baselines.get("spilu") or {}
    identity = baselines.get("identity") or {}
    jacobi_iter = jacobi.get("iterations")
    best_iter = best.get("iterations")
    if jacobi_iter and best_iter and success(best):
        iter_reduction = float(jacobi_iter) / max(1.0, float(best_iter))
    else:
        iter_reduction = None
    row = {
        "source_type": "training_meta",
        "run_dir": str(run_dir),
        "run_name": run_name,
        "mode": mode,
        "tier": infer_tier(matrix, run_name),
        "matrix": matrix,
        "best_epoch": meta.get("best_epoch"),
        "neuro_success": success(best),
        "jacobi_success": success(jacobi),
        "spilu_success": success(spilu),
        "rescued_from_jacobi_failure": (not success(jacobi)) and success(best),
        "iter_reduction_vs_jacobi": iter_reduction,
    }
    row.update(config)
    row.update(result_fields("best_neuro", best))
    row.update(result_fields("latest_neuro", latest))
    row.update(result_fields("jacobi", jacobi))
    row.update(result_fields("identity", identity))
    row.update(result_fields("spilu", spilu))
    return row


def summarize_end2end_csv(path):
    rows = []
    path = pathlib.Path(path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for item in reader:
            matrix = item.get("matrix")
            neuro_success = csv_success_from_iter(item, "neuro")
            jacobi_success = csv_success_from_iter(item, "jacobi")
            spilu_success = csv_success_from_iter(item, "spilu")
            row = {
                "source_type": "end2end_csv",
                "run_dir": str(path),
                "run_name": path.name,
                "mode": "end_to_end_timing",
                "tier": infer_tier(matrix, path.name),
                "matrix": matrix,
                "best_epoch": None,
                "neuro_success": neuro_success,
                "jacobi_success": jacobi_success,
                "spilu_success": spilu_success,
                "rescued_from_jacobi_failure": (jacobi_success is False) and (neuro_success is True),
                "iter_reduction_vs_jacobi": (
                    float(item["jacobi_iter"]) / max(1.0, float(item["neuro_iter"]))
                    if item.get("jacobi_iter") and item.get("neuro_iter") else None
                ),
                "topology_hop": None,
                "topology_row_topk": None,
                "amg_levels": None,
                "base_mode": item.get("base_mode"),
                "target_scale_mode": None,
                "hutchinson_probe_mode": None,
                "poison_train_replicas": None,
                "hidden_dim": None,
                "num_iterations": None,
                "best_neuro_iter": item.get("neuro_iter"),
                "best_neuro_info": item.get("neuro_info"),
                "best_neuro_final_rel": item.get("neuro_final_rel"),
                "best_neuro_solve_sec": item.get("neuro_solve_sec"),
                "latest_neuro_iter": None,
                "latest_neuro_info": None,
                "latest_neuro_final_rel": None,
                "latest_neuro_solve_sec": None,
                "jacobi_iter": item.get("jacobi_iter"),
                "jacobi_info": None,
                "jacobi_final_rel": None,
                "jacobi_solve_sec": item.get("jacobi_solve_sec"),
                "identity_iter": None,
                "identity_info": None,
                "identity_final_rel": None,
                "identity_solve_sec": None,
                "spilu_iter": item.get("spilu_iter"),
                "spilu_info": None,
                "spilu_final_rel": None,
                "spilu_solve_sec": item.get("spilu_solve_sec"),
                "neuro_strict_total_warm_sec": item.get("neuro_strict_total_warm_sec"),
                "spilu_total_sec": item.get("spilu_total_sec"),
                "graph_build_sec": item.get("graph_build_sec"),
                "forward_warm_sec": item.get("forward_warm_sec"),
                "g_build_sec": item.get("g_build_sec"),
            }
            rows.append(row)
    return rows


def write_rows(rows, out):
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "source_type", "mode", "tier", "matrix", "run_name", "best_epoch",
        "neuro_success", "jacobi_success", "spilu_success", "rescued_from_jacobi_failure",
        "best_neuro_iter", "best_neuro_info", "best_neuro_final_rel",
        "latest_neuro_iter", "latest_neuro_info", "latest_neuro_final_rel",
        "jacobi_iter", "jacobi_info", "jacobi_final_rel",
        "spilu_iter", "spilu_info", "spilu_final_rel",
        "iter_reduction_vs_jacobi", "neuro_strict_total_warm_sec", "spilu_total_sec",
        "topology_hop", "topology_row_topk", "amg_levels", "base_mode",
        "hutchinson_probe_mode", "poison_train_replicas", "run_dir",
    ]
    fieldnames = [f for f in preferred if f in fieldnames] + [
        f for f in fieldnames if f not in preferred
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def main():
    args = parse_args()
    rows = []
    for run_dir in args.run_dirs:
        rows.extend(summarize_run_dir(run_dir))
    for csv_path in args.end2end_csvs:
        rows.extend(summarize_end2end_csv(csv_path))
    if not rows:
        raise SystemExit("no rows to summarize")
    out = args.out or f"results/evaluation_summary/summary_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    out = write_rows(rows, out)
    print(f"wrote {out}")
    for row in rows:
        print(json.dumps({
            "mode": row.get("mode"),
            "tier": row.get("tier"),
            "matrix": row.get("matrix"),
            "best_neuro_iter": row.get("best_neuro_iter"),
            "best_neuro_info": row.get("best_neuro_info"),
            "neuro_success": row.get("neuro_success"),
        }), flush=True)


if __name__ == "__main__":
    main()
