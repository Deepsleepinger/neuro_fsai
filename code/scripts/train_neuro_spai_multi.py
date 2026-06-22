import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_neuro_spai_single import (
    NeuroSPAI,
    build_graph_tensors,
    build_spilu_inverse_row_topk,
    evaluate_values,
    load_rhs,
    metric_key,
    parse_float_list,
    run_bicgstab,
)
from utils.convert_suitesparse import canonicalize_sparse_matrix, read_mtx_from_tar


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one shared Neuro-SPAI model on multiple SPILU inverse top-k targets.")
    parser.add_argument("--matrix-tars", nargs="+", required=True)
    parser.add_argument("--save-dir", default="results/local_checkpoints")
    parser.add_argument("--exp-name", default="multi_neuro_spai")
    parser.add_argument("--row-topk", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-iterations", type=int, default=0)
    parser.add_argument("--weight-abs", type=float, default=5.0)
    parser.add_argument("--eval-damping-grid", default="1.0,0.5,0.25,0.1,0.05,0.01,0.005,0.001")
    parser.add_argument("--val-freq", type=int, default=100)
    parser.add_argument("--log-freq", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rhs-seed", type=int, default=20260622)
    parser.add_argument("--spilu-drop-tol", type=float, default=1e-4)
    parser.add_argument("--spilu-fill-factor", type=float, default=10.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--rtol", type=float, default=1e-8)
    return parser.parse_args()


def matrix_name(path):
    return pathlib.Path(path).name.replace(".tar.gz", "")


def build_case(tar_path, args, device):
    matrix = canonicalize_sparse_matrix(read_mtx_from_tar(tar_path))
    scale = max(1.0, float(np.max(np.abs(matrix.data))) if matrix.nnz else 1.0)
    A_scaled = (matrix / scale).astype(np.float64).tocsr()
    b = load_rhs(None, scale, A_scaled, args.rhs_seed)
    target_csr, teacher, dense_inverse = build_spilu_inverse_row_topk(
        A_scaled, args.row_topk, args.spilu_drop_tol, args.spilu_fill_factor)
    graph = build_graph_tensors(A_scaled, target_csr, device)
    diag = graph["safe_diag"]
    baselines = {
        "identity": run_bicgstab(A_scaled, b, lambda v: v, args.max_iter, args.rtol),
        "jacobi": run_bicgstab(A_scaled, b, lambda v: v / diag, args.max_iter, args.rtol),
        "spilu": run_bicgstab(A_scaled, b, teacher.solve, args.max_iter, args.rtol),
    }
    target_result, _ = evaluate_values(
        A_scaled, b, graph["target_values"], graph, args.max_iter, args.rtol)
    baselines["target_spai"] = target_result
    return {
        "name": matrix_name(tar_path),
        "tar_path": tar_path,
        "A_scaled": A_scaled,
        "b": b,
        "graph": graph,
        "weights": 1.0 + args.weight_abs * graph["target_norm"].abs(),
        "baselines": baselines,
        "target_nnz": int(target_csr.nnz),
        "inverse_absmax": float(np.abs(dense_inverse).max()),
        "scale": float(scale),
    }


def evaluate_case(model, case, damping_grid, args):
    graph = case["graph"]
    model.eval()
    with torch.no_grad():
        pred_norm = model(graph["node_attr"], graph["edge_index"], graph["edge_attr"])
        pred_residual = pred_norm.detach().cpu().numpy().astype(np.float64) * graph["target_scale"]
    best_result = None
    best_values = None
    best_alpha = None
    for alpha in damping_grid:
        values = graph["base_values64"] + alpha * pred_residual
        result, _ = evaluate_values(
            case["A_scaled"], case["b"], values, graph, args.max_iter, args.rtol)
        if best_result is None or metric_key(result) < metric_key(best_result):
            best_result = result
            best_values = values
            best_alpha = alpha
    return best_result, best_alpha, best_values


def aggregate_metric(results_by_name, cases):
    ratios = []
    for case in cases:
        result = results_by_name[case["name"]]["result"]
        jacobi_iter = max(1, case["baselines"]["jacobi"]["iterations"])
        effective_iter = result["iterations"] if result["info"] == 0 else 10 * jacobi_iter
        ratios.append(effective_iter / jacobi_iter)
    return float(np.mean(ratios))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_dir = pathlib.Path(args.save_dir) / f"{args.exp_name}-rowtop{args.row_topk}-{time.strftime('%Y%m%d-%H%M%S')}"
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"

    def log(message):
        print(message, flush=True)
        with log_path.open("a") as f:
            f.write(message + "\n")

    log(f"device={device}")
    log(f"run_dir={run_dir}")
    cases = []
    for tar_path in args.matrix_tars:
        start = time.perf_counter()
        case = build_case(tar_path, args, device)
        cases.append(case)
        log(
            f"case={case['name']} N={case['A_scaled'].shape[0]} "
            f"A_nnz={case['A_scaled'].nnz} target_nnz={case['target_nnz']} "
            f"target_scale={case['graph']['target_scale']:.6e} "
            f"build_time={time.perf_counter() - start:.3f}s "
            f"baselines={json.dumps(case['baselines'])}")

    first = cases[0]["graph"]
    model = NeuroSPAI(
        node_dim=first["node_attr"].shape[1],
        edge_dim=first["edge_attr"].shape[1],
        hidden_dim=args.hidden_dim,
        num_iterations=args.num_iterations,
        use_node_embedding=False,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    damping_grid = parse_float_list(args.eval_damping_grid)

    best_metric = float("inf")
    best_epoch = -1
    best_results = None
    latest_results = None
    for epoch in range(args.epochs + 1):
        total_loss = 0.0
        total_mae = 0.0
        for case in cases:
            model.train()
            graph = case["graph"]
            pred_norm = model(graph["node_attr"], graph["edge_index"], graph["edge_attr"])
            sq = (pred_norm - graph["target_norm"]).pow(2)
            loss = (sq * case["weights"]).mean()
            mae = (pred_norm - graph["target_norm"]).abs().mean()
            if epoch > 0:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            total_mae += float(mae.item())

        should_eval = epoch == 0 or epoch == args.epochs or epoch % args.val_freq == 0
        if should_eval:
            results = {}
            for case in cases:
                result, alpha, values = evaluate_case(model, case, damping_grid, args)
                results[case["name"]] = {"result": result, "alpha": alpha}
                if best_results is None:
                    continue
            metric = aggregate_metric(results, cases)
            latest_results = results
            if metric < best_metric:
                best_metric = metric
                best_epoch = epoch
                best_results = results
                torch.save(model.state_dict(), model_dir / "best_val.pt")
            compact = {
                name: {
                    "iter": row["result"]["iterations"],
                    "info": row["result"]["info"],
                    "alpha": row["alpha"],
                }
                for name, row in results.items()
            }
            log(
                f"epoch={epoch:04d} loss={total_loss / len(cases):.6e} "
                f"mae={total_mae / len(cases):.6e} metric={metric:.6e} "
                f"best_epoch={best_epoch} results={json.dumps(compact)}")
        elif epoch % args.log_freq == 0:
            log(
                f"epoch={epoch:04d} loss={total_loss / len(cases):.6e} "
                f"mae={total_mae / len(cases):.6e} best_epoch={best_epoch}")

    torch.save(model.state_dict(), model_dir / "latest_model.pt")
    meta = {
        "args": vars(args),
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_results": best_results,
        "latest_results": latest_results,
        "cases": [
            {
                "name": case["name"],
                "tar_path": case["tar_path"],
                "N": int(case["A_scaled"].shape[0]),
                "A_nnz": int(case["A_scaled"].nnz),
                "target_nnz": case["target_nnz"],
                "target_scale": float(case["graph"]["target_scale"]),
                "baselines": case["baselines"],
            }
            for case in cases
        ],
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log("summary=" + json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
