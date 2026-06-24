import argparse
import json
import pathlib
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import LinearOperator, bicgstab, spilu, spsolve_triangular
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model_neuro_ilu import Net
from pcg import build_neuro_ilu_csr, edge_to_csr
from utils.training_utils_neuro_ilu import train_epoch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset/diffusivity_100.0/circle_low_res")
    parser.add_argument("--train-files", type=int, default=2)
    parser.add_argument("--eval-files", type=int, default=1)
    parser.add_argument("--train-samples-per-file", type=int, default=4)
    parser.add_argument("--eval-samples-per-file", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--num-iterations", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--output", default="results/heat_bicgstab_comparison.json")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_graph(sample):
    return Data(
        x=torch.as_tensor(sample["x"]).float(),
        edge_attr=torch.as_tensor(sample["edge_attr"]).float(),
        edge_index=torch.as_tensor(sample["edge_index"]).long(),
        y=torch.as_tensor(sample["rhs"]).float(),
        u=torch.as_tensor(sample["u"]).float(),
        diag=torch.as_tensor(sample["diag"]).reshape(-1, 1).float(),
        r=torch.as_tensor(sample["r"]).float(),
        rhs=torch.as_tensor(sample["rhs"]).float(),
        u_next=torch.as_tensor(sample["u_next"]).float(),
    )


def load_split(data_dir, train_files, eval_files, train_samples_per_file, eval_samples_per_file):
    files = sorted(pathlib.Path(data_dir).glob("*.npy"))
    if len(files) < train_files + eval_files:
        raise ValueError("not enough data shards for requested split")

    train_graphs = []
    eval_graphs = []
    for file_path in files[:train_files]:
        data = np.load(file_path, allow_pickle=True)
        for sample in data[:train_samples_per_file]:
            train_graphs.append(make_graph(sample))

    for file_path in files[train_files:train_files + eval_files]:
        data = np.load(file_path, allow_pickle=True)
        for sample in data[:eval_samples_per_file]:
            eval_graphs.append(make_graph(sample))

    return train_graphs, eval_graphs


def build_model_args(hidden_dim, num_iterations):
    return SimpleNamespace(
        dataset="heatmultisource",
        hidden_dim=hidden_dim,
        hidden_layers_encoder=1,
        hidden_layers_decoder=1,
        hidden_layers_processor=1,
        num_iterations=num_iterations,
        norm="LayerNorm",
        use_global=False,
        use_r=False,
        diagonalize=False,
        use_pred_x=False,
        x_loss_weight=0.0,
        frob_loss_weight=1.0,
    )


def build_model(model_args, graph):
    return Net(
        model_args,
        in_dim_node=graph.x.shape[-1],
        in_dim_edge=graph.edge_attr.shape[-1],
        out_dim=1,
        b_dim=graph.x.shape[0],
        num_edges=graph.edge_attr.shape[0],
        out_dim_node=model_args.hidden_dim,
        out_dim_edge=model_args.hidden_dim,
        hidden_dim_node=model_args.hidden_dim,
        hidden_dim_edge=model_args.hidden_dim,
        hidden_layers_node=model_args.hidden_layers_encoder,
        hidden_layers_edge=model_args.hidden_layers_encoder,
        num_iterations=model_args.num_iterations,
        hidden_dim_processor_node=model_args.hidden_dim,
        hidden_dim_processor_edge=model_args.hidden_dim,
        hidden_layers_processor_node=model_args.hidden_layers_processor,
        hidden_layers_processor_edge=model_args.hidden_layers_processor,
        hidden_dim_decoder=model_args.hidden_dim,
        hidden_layers_decoder=model_args.hidden_layers_decoder,
        dirichlet_idx=3,
        norm_type=model_args.norm,
    )


def apply_dirichlet_to_system(A_csr, b_np, dirichlet_mask):
    if not dirichlet_mask.any().item():
        return A_csr, b_np

    A_dense = A_csr.toarray()
    b_mod = b_np.copy()
    for dn in torch.where(dirichlet_mask)[0].cpu().numpy():
        A_dense[dn, :] = 0.0
        A_dense[:, dn] = 0.0
        A_dense[dn, dn] = 1.0
        b_mod[dn] = 0.0
    return csr_matrix(A_dense), b_mod


def run_scipy_bicgstab(A_csr, b_np, matvec, max_iter, rtol):
    counter = {"iters": 0}

    def callback(_xk):
        counter["iters"] += 1

    M = LinearOperator(A_csr.shape, matvec=matvec, dtype=np.float64)
    start = time.perf_counter()
    _x, info = bicgstab(A_csr, b_np, x0=np.zeros(A_csr.shape[0]), rtol=rtol, atol=0.0, maxiter=max_iter, M=M, callback=callback)
    solve_time = time.perf_counter() - start
    iterations = counter["iters"] if info == 0 else max_iter
    return iterations, solve_time


def evaluate_model(model, eval_graphs, device, max_iter, rtol):
    model.eval()
    metrics = {name: [] for name in ["neuro", "ilu0", "jacobi", "identity"]}

    with torch.no_grad():
        for graph in eval_graphs:
            data = graph.to(device)
            dirichlet_mask = data.x[:, 3].to(torch.bool)
            N = data.x.shape[0]

            setup_start = time.perf_counter()
            _, _, (L_ei, L_val), (U_ei, U_val), (A_ei, A_val) = model(
                data.x,
                data.edge_attr,
                data.edge_index,
                diag=data.diag,
                input_r=data.r,
                input_x=torch.zeros_like(data.u_next),
                batch_idx=torch.zeros(data.x.shape[0], dtype=torch.long, device=device),
                include_r=False,
                use_global=False,
                diagonalize=False,
                use_pred_x=False,
            )
            A_csr = edge_to_csr(A_ei, A_val, N)
            b_np = data.rhs.detach().cpu().numpy().ravel().astype(np.float64)
            A_csr, b_np = apply_dirichlet_to_system(A_csr, b_np, dirichlet_mask)
            L_csr, U_csr = build_neuro_ilu_csr(L_ei, L_val, U_ei, U_val, N, dirichlet_mask=dirichlet_mask)
            neuro_setup = time.perf_counter() - setup_start

            def neuro_matvec(r):
                w = spsolve_triangular(L_csr, r, lower=True, unit_diagonal=True)
                return spsolve_triangular(U_csr, w, lower=False, unit_diagonal=False)

            neuro_iter, neuro_solve = run_scipy_bicgstab(A_csr, b_np, neuro_matvec, max_iter, rtol)
            metrics["neuro"].append({
                "iterations": neuro_iter,
                "setup_time": neuro_setup,
                "solve_time": neuro_solve,
                "total_time": neuro_setup + neuro_solve,
            })

            ilu_setup_start = time.perf_counter()
            ilu = spilu(A_csr.tocsc(), drop_tol=0.0, fill_factor=1.0)
            ilu_setup = time.perf_counter() - ilu_setup_start
            ilu_iter, ilu_solve = run_scipy_bicgstab(A_csr, b_np, ilu.solve, max_iter, rtol)
            metrics["ilu0"].append({
                "iterations": ilu_iter,
                "setup_time": ilu_setup,
                "solve_time": ilu_solve,
                "total_time": ilu_setup + ilu_solve,
            })

            diag_vals = A_csr.diagonal().astype(np.float64)
            diag_vals[diag_vals == 0.0] = 1.0
            jacobi_setup_start = time.perf_counter()
            jacobi_diag = 1.0 / diag_vals
            jacobi_setup = time.perf_counter() - jacobi_setup_start
            jacobi_iter, jacobi_solve = run_scipy_bicgstab(A_csr, b_np, lambda r: jacobi_diag * r, max_iter, rtol)
            metrics["jacobi"].append({
                "iterations": jacobi_iter,
                "setup_time": jacobi_setup,
                "solve_time": jacobi_solve,
                "total_time": jacobi_setup + jacobi_solve,
            })

            ident_setup_start = time.perf_counter()
            ident_setup = time.perf_counter() - ident_setup_start
            ident_iter, ident_solve = run_scipy_bicgstab(A_csr, b_np, lambda r: r, max_iter, rtol)
            metrics["identity"].append({
                "iterations": ident_iter,
                "setup_time": ident_setup,
                "solve_time": ident_solve,
                "total_time": ident_setup + ident_solve,
            })

    return metrics


def summarize(metric_rows):
    summary = {}
    for name, rows in metric_rows.items():
        summary[name] = {
            "count": len(rows),
            "avg_iterations": float(np.mean([r["iterations"] for r in rows])),
            "avg_setup_time": float(np.mean([r["setup_time"] for r in rows])),
            "avg_solve_time": float(np.mean([r["solve_time"] for r in rows])),
            "avg_total_time": float(np.mean([r["total_time"] for r in rows])),
            "median_iterations": float(np.median([r["iterations"] for r in rows])),
        }

    neuro_total = summary["neuro"]["avg_total_time"]
    for baseline in ["ilu0", "jacobi", "identity"]:
        baseline_total = summary[baseline]["avg_total_time"]
        summary[baseline]["total_time_ratio_vs_neuro"] = float(baseline_total / neuro_total) if neuro_total > 0 else float("nan")
        summary[baseline]["neuro_total_time_divided_by_baseline"] = float(neuro_total / baseline_total) if baseline_total > 0 else float("nan")

    return summary


def main():
    args = parse_args()
    set_seed(args.seed)

    train_graphs, eval_graphs = load_split(
        args.data_dir,
        args.train_files,
        args.eval_files,
        args.train_samples_per_file,
        args.eval_samples_per_file,
    )

    model_args = build_model_args(args.hidden_dim, args.num_iterations)
    model = build_model(model_args, train_graphs[0])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    train_args = SimpleNamespace(
        use_r=False,
        use_global=False,
        diagonalize=False,
        use_pred_x=False,
        x_loss_weight=0.0,
        frob_loss_weight=1.0,
    )

    total_steps = 0
    epoch_losses = []
    for epoch in range(args.epochs):
        avg_loss, total_steps = train_epoch(
            train_args,
            train_loader,
            model,
            optimizer,
            scheduler,
            device,
            tb_writer=None,
            total_steps=total_steps,
            epoch=epoch,
            logger=None,
        )
        epoch_losses.append(float(avg_loss))

    metrics = evaluate_model(model, eval_graphs, device, args.max_iter, args.rtol)
    summary = summarize(metrics)
    report = {
        "config": vars(args),
        "device": str(device),
        "train_graphs": len(train_graphs),
        "eval_graphs": len(eval_graphs),
        "epoch_losses": epoch_losses,
        "summary": summary,
    }

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
