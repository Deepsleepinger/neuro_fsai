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


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_neuro_ilu_suitesparse import build_model
from utils.convert_suitesparse import canonicalize_sparse_matrix, matrix_to_graph
from utils.training_utils_neuro_ilu import frobenius_loss as fast_frobenius_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/mnt/h/neuro_ilu/checkpoints/suitesparse_run3-neuroilu-20260611-164503/model/best_val.pt")
    parser.add_argument("--selected-json", default="results/selected_eval_matrices.json")
    parser.add_argument("--limit-per-scale", type=int, default=1)
    parser.add_argument("--max-entries", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-json", default="results/frobenius_loss_profile.json")
    parser.add_argument("--output-md", default="results/frobenius_loss_profile.md")
    return parser.parse_args()


def read_matrix_from_tar(tar_path):
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".mtx"):
                fh = tar.extractfile(member)
                if fh is not None:
                    return mmread(io.BytesIO(fh.read()))
    raise FileNotFoundError(f"no .mtx found in {tar_path}")


def sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _subsample_pattern(edge_index, values, max_entries, deterministic=False):
    if max_entries is None or max_entries < 0 or edge_index.shape[1] <= max_entries:
        return edge_index, values

    num_entries = edge_index.shape[1]
    device = edge_index.device
    if deterministic:
        sample_idx = (torch.arange(max_entries, device=device, dtype=torch.long) * num_entries) // max_entries
    else:
        sample_idx = torch.randint(num_entries, (max_entries,), device=device)

    return edge_index[:, sample_idx], values[sample_idx]


def _segment_ptr(index, n):
    counts = torch.bincount(index, minlength=n)
    ptr = torch.zeros(n + 1, device=index.device, dtype=torch.long)
    ptr[1:] = counts.cumsum(0)
    return ptr


def _matched_dot(left_keys, left_values, right_keys, right_values):
    pos = torch.searchsorted(right_keys, left_keys)
    valid = pos < right_keys.numel()
    if not valid.any():
        return left_values.new_zeros(())

    left_keys = left_keys[valid]
    left_values = left_values[valid]
    pos = pos[valid]
    matched = right_keys[pos] == left_keys
    if not matched.any():
        return left_values.new_zeros(())

    return (left_values[matched] * right_values[pos[matched]]).sum()


def old_loop_frobenius_loss(L_edge_index, L_values, U_edge_index, U_values,
                            A_edge_index, A_values, n, dirichlet_mask=None,
                            max_entries=None, deterministic=False):
    A_edge_index, A_values = _subsample_pattern(
        A_edge_index, A_values, max_entries=max_entries, deterministic=deterministic)

    l_order = torch.argsort(L_edge_index[0] * n + L_edge_index[1])
    l_rows = L_edge_index[0][l_order]
    l_cols = L_edge_index[1][l_order]
    l_vals = L_values[l_order]
    l_ptr = _segment_ptr(l_rows, n)

    u_order = torch.argsort(U_edge_index[1] * n + U_edge_index[0])
    u_cols = U_edge_index[1][u_order]
    u_rows = U_edge_index[0][u_order]
    u_vals = U_values[u_order]
    u_ptr = _segment_ptr(u_cols, n)

    sampled = []
    zero = L_values.new_zeros(())
    query_rows = A_edge_index[0]
    query_cols = A_edge_index[1]

    for idx in range(A_edge_index.shape[1]):
        row_i = int(query_rows[idx].item())
        col_j = int(query_cols[idx].item())
        l_start = int(l_ptr[row_i].item())
        l_end = int(l_ptr[row_i + 1].item())
        u_start = int(u_ptr[col_j].item())
        u_end = int(u_ptr[col_j + 1].item())

        if l_start == l_end or u_start == u_end:
            sampled.append(zero)
            continue

        row_cols = l_cols[l_start:l_end]
        row_vals = l_vals[l_start:l_end]
        col_rows = u_rows[u_start:u_end]
        col_vals = u_vals[u_start:u_end]

        if row_cols.numel() <= col_rows.numel():
            sampled.append(_matched_dot(row_cols, row_vals, col_rows, col_vals))
        else:
            sampled.append(_matched_dot(col_rows, col_vals, row_cols, row_vals))

    pred_values = torch.stack(sampled) if sampled else L_values.new_zeros((0,))

    if dirichlet_mask is not None:
        active_mask = ~(dirichlet_mask[A_edge_index[0]] | dirichlet_mask[A_edge_index[1]])
        pred_values = pred_values[active_mask]
        A_values = A_values[active_mask]

    if pred_values.numel() == 0:
        loss = L_values.new_zeros(())
    else:
        diff = pred_values - A_values
        loss = (diff ** 2).mean()

    reg_L = (L_values ** 2).mean() * 1e-6
    reg_U = (U_values ** 2).mean() * 1e-6
    return loss + reg_L + reg_U


def prepare_case(model, matrix_path, device):
    matrix = canonicalize_sparse_matrix(read_matrix_from_tar(matrix_path))
    graph = matrix_to_graph(matrix, name=pathlib.Path(matrix_path).stem.replace(".tar", ""))

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

    with torch.no_grad():
        _, _, (L_ei, L_val), (U_ei, U_val), (A_ei, A_val) = model(
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

    dirichlet_mask = x[:, 3].to(torch.bool)
    return {
        "name": graph["meta"]["name"],
        "n": int(x.shape[0]),
        "nnz_A": int(A_ei.shape[1]),
        "nnz_L": int(L_ei.shape[1]),
        "nnz_U": int(U_ei.shape[1]),
        "L_ei": L_ei,
        "L_val": L_val,
        "U_ei": U_ei,
        "U_val": U_val,
        "A_ei": A_ei,
        "A_val": A_val,
        "dirichlet_mask": dirichlet_mask,
    }


def benchmark_loss(loss_fn, case, max_entries, repeats, warmup, device):
    def run_once():
        L_val = case["L_val"].detach().clone().requires_grad_(True)
        U_val = case["U_val"].detach().clone().requires_grad_(True)

        sync_if_needed(device)
        t0 = time.perf_counter()
        loss = loss_fn(
            case["L_ei"], L_val,
            case["U_ei"], U_val,
            case["A_ei"], case["A_val"],
            case["n"],
            dirichlet_mask=case["dirichlet_mask"],
            max_entries=max_entries,
            deterministic=True)
        sync_if_needed(device)
        t1 = time.perf_counter()
        loss.backward()
        sync_if_needed(device)
        t2 = time.perf_counter()
        return {
            "loss": float(loss.detach().cpu().item()),
            "forward_ms": (t1 - t0) * 1000.0,
            "backward_ms": (t2 - t1) * 1000.0,
            "total_ms": (t2 - t0) * 1000.0,
        }

    for _ in range(max(0, warmup)):
        run_once()

    rows = [run_once() for _ in range(repeats)]
    return {
        "loss": float(np.mean([r["loss"] for r in rows])),
        "forward_ms": float(np.mean([r["forward_ms"] for r in rows])),
        "backward_ms": float(np.mean([r["backward_ms"] for r in rows])),
        "total_ms": float(np.mean([r["total_ms"] for r in rows])),
        "runs": rows,
    }


def format_markdown(report):
    lines = []
    lines.append("# Frobenius Loss Kernel Profiling")
    lines.append("")
    lines.append(f"- Checkpoint: `{report['checkpoint']}`")
    lines.append(f"- Device: `{report['device']}`")
    lines.append(f"- Max entries: `{report['max_entries']}`")
    lines.append(f"- Repeats: `{report['repeats']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| Avg old total (ms) | {report['summary']['avg_old_total_ms']:.3f} |")
    lines.append(f"| Avg new total (ms) | {report['summary']['avg_new_total_ms']:.3f} |")
    lines.append(f"| Avg speedup | {report['summary']['avg_total_speedup']:.2f}x |")
    lines.append(f"| Avg forward speedup | {report['summary']['avg_forward_speedup']:.2f}x |")
    lines.append(f"| Avg backward speedup | {report['summary']['avg_backward_speedup']:.2f}x |")
    lines.append(f"| Max abs loss diff | {report['summary']['max_abs_loss_diff']:.6e} |")
    lines.append("")
    lines.append("## Per Matrix")
    lines.append("")
    lines.append("| Matrix | Scale | N | nnz(A) | Old Total (ms) | New Total (ms) | Speedup | Loss Diff |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in report["results"]:
        lines.append(
            f"| {row['name']} | {row['scale']} | {row['n']} | {row['nnz_A']} | "
            f"{row['old']['total_ms']:.3f} | {row['new']['total_ms']:.3f} | "
            f"{row['speedup_total']:.2f}x | {row['abs_loss_diff']:.3e} |")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    selected = json.loads(pathlib.Path(args.selected_json).read_text())
    chosen = []
    for scale, items in selected["scales"].items():
        chosen.extend((scale, item) for item in items[:args.limit_per_scale])

    model = build_model().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    results = []
    for scale, item in chosen:
        case = prepare_case(model, item["path"], device)
        old_stats = benchmark_loss(
            old_loop_frobenius_loss, case, args.max_entries, args.repeats, args.warmup, device)
        new_stats = benchmark_loss(
            fast_frobenius_loss, case, args.max_entries, args.repeats, args.warmup, device)

        results.append({
            "name": case["name"],
            "scale": scale,
            "path": item["path"],
            "n": case["n"],
            "nnz_A": case["nnz_A"],
            "nnz_L": case["nnz_L"],
            "nnz_U": case["nnz_U"],
            "old": old_stats,
            "new": new_stats,
            "abs_loss_diff": abs(old_stats["loss"] - new_stats["loss"]),
            "speedup_total": old_stats["total_ms"] / max(new_stats["total_ms"], 1e-12),
            "speedup_forward": old_stats["forward_ms"] / max(new_stats["forward_ms"], 1e-12),
            "speedup_backward": old_stats["backward_ms"] / max(new_stats["backward_ms"], 1e-12),
        })

    summary = {
        "avg_old_total_ms": float(np.mean([r["old"]["total_ms"] for r in results])),
        "avg_new_total_ms": float(np.mean([r["new"]["total_ms"] for r in results])),
        "avg_total_speedup": float(np.mean([r["speedup_total"] for r in results])),
        "avg_forward_speedup": float(np.mean([r["speedup_forward"] for r in results])),
        "avg_backward_speedup": float(np.mean([r["speedup_backward"] for r in results])),
        "max_abs_loss_diff": float(np.max([r["abs_loss_diff"] for r in results])),
    }

    report = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "max_entries": args.max_entries,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "results": results,
        "summary": summary,
    }

    output_json = ROOT / args.output_json
    output_md = ROOT / args.output_md
    output_json.write_text(json.dumps(report, indent=2))
    output_md.write_text(format_markdown(report))

    print(json.dumps(summary, indent=2))
    print(f"saved json to {output_json}")
    print(f"saved md to {output_md}")


if __name__ == "__main__":
    main()
