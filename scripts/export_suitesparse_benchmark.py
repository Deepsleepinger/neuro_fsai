import argparse
import csv
import json
from pathlib import Path


DEFAULT_METHODS = ["neuro", "neuro_diagfix", "neuro_fsai", "ilu0", "spilu", "jacobi", "identity"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--markdown-out", required=True)
    parser.add_argument("--csv-out", required=True)
    return parser.parse_args()


def fmt(x, digits=6):
    if x is None:
        return ""
    if isinstance(x, int):
        return str(x)
    return f"{x:.{digits}f}"


def report_methods(report):
    rows = report.get("results", [])
    return [m for m in DEFAULT_METHODS if any(m in row for row in rows)]


def successful_methods(row, methods):
    return {m: row[m] for m in methods if m in row and row[m]["info"] == 0}


def best_method(row, key, methods):
    succ = successful_methods(row, methods)
    if not succ:
        return ""
    return min(succ.items(), key=lambda kv: kv[1][key])[0]


def write_csv(report, path):
    methods = report_methods(report)
    fieldnames = ["name", "scale", "nrows", "nnz"]
    for method in methods:
        fieldnames.extend([
            f"{method}_iterations",
            f"{method}_setup_time",
            f"{method}_solve_time",
            f"{method}_total_time",
            f"{method}_info",
            f"{method}_error",
        ])
    fieldnames.extend(["best_iterations_method", "best_total_time_method"])

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["results"]:
            out = {
                "name": row["name"],
                "scale": row["scale"],
                "nrows": row["nrows"],
                "nnz": row["nnz"],
                "best_iterations_method": best_method(row, "iterations", methods),
                "best_total_time_method": best_method(row, "total_time", methods),
            }
            for method in methods:
                data = row.get(method, {})
                out[f"{method}_iterations"] = data.get("iterations", "")
                out[f"{method}_setup_time"] = data.get("setup_time", "")
                out[f"{method}_solve_time"] = data.get("solve_time", "")
                out[f"{method}_total_time"] = data.get("total_time", "")
                out[f"{method}_info"] = data.get("info", "")
                out[f"{method}_error"] = data.get("error", "")
            writer.writerow(out)


def build_markdown(report):
    methods = report_methods(report)
    lines = []
    lines.append("# SuiteSparse Benchmark Summary")
    lines.append("")
    lines.append(f"- Checkpoint: `{report['checkpoint']}`")
    if "selected_json" in report:
        lines.append(f"- Selected JSON: `{report['selected_json']}`")
    lines.append(f"- Device: `{report['device']}`")
    if "max_iter" in report:
        lines.append(f"- Solver: `bicgstab`, rtol=`{report.get('rtol')}`, max_iter=`{report.get('max_iter')}`")
    if "spilu_drop_tol" in report:
        lines.append(
            f"- SPILU: drop_tol=`{report.get('spilu_drop_tol')}`, "
            f"fill_factor=`{report.get('spilu_fill_factor')}`"
        )
    lines.append(f"- Matrix count: `{report['matrix_count']}`")
    lines.append("")

    lines.append("## Overall Summary")
    lines.append("")
    lines.append("| Method | Success | Failure | Avg Iter | Median Iter | Avg Setup (s) | Avg Solve (s) | Avg Total (s) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    overall = report["summary"]["overall"]
    for method in methods:
        data = overall[method]
        lines.append(
            f"| {method} | {data['success_count']} | {data['failure_count']} | "
            f"{fmt(data['avg_iterations'], 2)} | {fmt(data['median_iterations'], 2)} | "
            f"{fmt(data['avg_setup_time'])} | {fmt(data['avg_solve_time'])} | {fmt(data['avg_total_time'])} |"
        )
    lines.append("")

    lines.append("## By Scale")
    lines.append("")
    for scale, block in report["summary"]["by_scale"].items():
        lines.append(f"### {scale}")
        lines.append("")
        lines.append("| Method | Success | Failure | Avg Iter | Avg Total (s) |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for method in methods:
            data = block[method]
            lines.append(
                f"| {method} | {data['success_count']} | {data['failure_count']} | "
                f"{fmt(data['avg_iterations'], 2)} | {fmt(data['avg_total_time'])} |"
            )
        lines.append("")

    lines.append("## Per-Matrix Winners")
    lines.append("")
    lines.append("| Matrix | Scale | Best Iterations | Best Total Time |")
    lines.append("| --- | --- | --- | --- |")
    for row in report["results"]:
        lines.append(
            f"| {row['name']} | {row['scale']} | "
            f"{best_method(row, 'iterations', methods) or 'none'} | {best_method(row, 'total_time', methods) or 'none'} |"
        )
    lines.append("")

    return "\n".join(lines)


def main():
    args = parse_args()
    report = json.loads(Path(args.input).read_text())

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(report, csv_path)

    markdown_path = Path(args.markdown_out)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_markdown(report))

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
