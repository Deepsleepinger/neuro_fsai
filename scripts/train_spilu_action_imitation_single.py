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
import torch.nn.functional as F
from scipy.io import mmread
from scipy.sparse.linalg import LinearOperator, bicgstab, spilu


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.suitesparse_dataset import SuiteSparseDataset
from models.model_neuro_fsai import Net
from pcg import build_neuro_fsai_csr
from utils.convert_suitesparse import canonicalize_sparse_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-tar", required=True)
    parser.add_argument("--train-data-dir", required=True)
    parser.add_argument("--val-data-dir", required=True)
    parser.add_argument("--save-dir", default="results/local_checkpoints")
    parser.add_argument("--exp-name", default="spilu_action_imitation_single")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-probes", type=int, default=8)
    parser.add_argument("--val-probes", type=int, default=32)
    parser.add_argument("--val-freq", type=int, default=25)
    parser.add_argument("--log-freq", type=int, default=25)
    parser.add_argument("--checkpoint-metric", choices=["loss", "solver"], default="loss")
    parser.add_argument("--solver-val-freq", type=int, default=25)
    parser.add_argument("--solver-val-max-iter", type=int, default=50)
    parser.add_argument("--solver-val-rtol", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--spilu-drop-tol", type=float, default=1e-4)
    parser.add_argument("--spilu-fill-factor", type=float, default=10.0)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--residual-clip", type=float, default=10.0)
    parser.add_argument("--reg-weight", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--hidden-layers-encoder", type=int, default=1)
    parser.add_argument("--hidden-layers-processor", type=int, default=1)
    parser.add_argument("--hidden-layers-decoder", type=int, default=1)
    parser.add_argument("--fsai-offdiag-scale", type=float, default=0.1)
    parser.add_argument("--fsai-offdiag-basis-cap", type=float, default=1.0)
    parser.add_argument("--fsai-diag-scale", type=float, default=0.0)
    parser.add_argument("--fsai-diag-abs-floor", type=float, default=1e-2)
    parser.add_argument("--fsai-jacobi-eps", type=float, default=1e-12)
    parser.add_argument("--fsai-relative-value-clip", type=float, default=10.0)
    return parser.parse_args()


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
        use_pred_x=False,
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


def rademacher(num_nodes, num_probes, device, dtype, generator):
    v = torch.randint(
        0, 2, (num_nodes, num_probes),
        device=device, dtype=torch.int64, generator=generator)
    return (v * 2 - 1).to(dtype=dtype)


def teacher_solve(spilu_factor, probes):
    rhs = probes.detach().cpu().numpy().astype(np.float64)
    solved = spilu_factor.solve(rhs)
    return torch.from_numpy(np.asarray(solved)).to(
        device=probes.device, dtype=probes.dtype)


def relative_huber(pred, target, huber_beta, residual_clip):
    denom = target.pow(2).mean().sqrt().clamp_min(1e-6)
    rel = (pred - target) / denom
    if residual_clip is not None and residual_clip > 0:
        rel = rel.clamp(min=-residual_clip, max=residual_clip)
    return F.smooth_l1_loss(
        rel, torch.zeros_like(rel), beta=huber_beta, reduction="mean")


def value_regularization(G_L_ei, G_L_val, G_U_ei, G_U_val):
    terms = []
    l_mask = G_L_ei[0] != G_L_ei[1]
    u_mask = G_U_ei[0] != G_U_ei[1]
    if l_mask.any():
        terms.append(G_L_val[l_mask].pow(2).mean())
    if u_mask.any():
        terms.append(G_U_val[u_mask].pow(2).mean())
    if not terms:
        return G_L_val.new_zeros(())
    return 1e-6 * torch.stack(terms).mean()


def forward_action(model, graph, probes, device):
    node_attr = graph.x.to(device)
    edge_attr = graph.edge_attr.to(device)
    edge_index = graph.edge_index.to(device)
    diag = graph.diag.to(device) if hasattr(graph, "diag") else None
    input_x = torch.zeros((node_attr.shape[0], 1), device=device, dtype=node_attr.dtype)
    _, pred, (G_L_ei, G_L_val), (G_U_ei, G_U_val), _ = model(
        node_attr, edge_attr, edge_index,
        diag=diag,
        input_r=probes,
        input_x=input_x,
        batch_idx=torch.zeros(node_attr.shape[0], dtype=torch.long, device=device),
        include_r=False,
        use_global=False,
        diagonalize=False,
        use_pred_x=False,
    )
    return pred, G_L_ei, G_L_val, G_U_ei, G_U_val


def solver_validation(model, graph, A_csr, device, max_iter, rtol):
    model.eval()
    node_attr = graph.x.to(device)
    edge_attr = graph.edge_attr.to(device)
    edge_index = graph.edge_index.to(device)
    diag = graph.diag.to(device) if hasattr(graph, "diag") else None
    input_x = torch.zeros((node_attr.shape[0], 1), device=device, dtype=node_attr.dtype)
    with torch.no_grad():
        _, _, (G_L_ei, G_L_val), (G_U_ei, G_U_val), _ = model(
            node_attr, edge_attr, edge_index,
            diag=diag,
            input_r=torch.zeros((node_attr.shape[0], 1), device=device, dtype=node_attr.dtype),
            input_x=input_x,
            batch_idx=torch.zeros(node_attr.shape[0], dtype=torch.long, device=device),
            include_r=False,
            use_global=False,
            diagonalize=False,
            use_pred_x=False,
        )
    G_L_csr, G_U_csr = build_neuro_fsai_csr(
        G_L_ei, G_L_val, G_U_ei, G_U_val, node_attr.shape[0],
        dirichlet_mask=torch.zeros(node_attr.shape[0], dtype=torch.bool))

    b = graph.rhs.detach().cpu().numpy().reshape(-1).astype(np.float64)
    norm0 = max(1e-30, float(np.linalg.norm(b)))
    residual_history = []

    def matvec(v):
        return G_U_csr @ (G_L_csr @ v)

    def callback(xk):
        rel = float(np.linalg.norm(b - A_csr @ xk) / norm0)
        residual_history.append(rel)

    M = LinearOperator(A_csr.shape, matvec=matvec, dtype=np.float64)
    x, info = bicgstab(
        A_csr, b, x0=np.zeros(A_csr.shape[0]),
        rtol=rtol, atol=0.0, maxiter=max_iter,
        M=M, callback=callback)
    final_rel = float(np.linalg.norm(b - A_csr @ x) / norm0)
    return {
        "metric": final_rel,
        "info": int(info),
        "iterations": len(residual_history) if info == 0 else max_iter,
        "final_rel": final_rel,
        "min_rel": min(residual_history + [final_rel]),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_dir = pathlib.Path(args.save_dir) / f"{args.exp_name}-spiluteacher-{time.strftime('%Y%m%d-%H%M%S')}"
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"

    def log(message):
        print(message, flush=True)
        with log_path.open("a") as f:
            f.write(message + "\n")

    train_ds = SuiteSparseDataset(args.train_data_dir, use_data_num=-1)
    val_ds = SuiteSparseDataset(args.val_data_dir, use_data_num=-1)
    if len(train_ds) != 1 or len(val_ds) != 1:
        raise ValueError("This script is intentionally single-graph only.")
    train_graph = train_ds[0]
    val_graph = val_ds[0]
    num_nodes = train_graph.x.shape[0]

    matrix = canonicalize_sparse_matrix(read_matrix_from_tar(args.matrix_tar))
    scale = max(1.0, float(np.max(np.abs(matrix.data))) if matrix.nnz else 1.0)
    A_scaled = (matrix / scale).tocsc()
    log(f"device={device}")
    log(f"run_dir={run_dir}")
    log(f"matrix={args.matrix_tar} N={A_scaled.shape[0]} nnz={A_scaled.nnz} scale={scale:.6e}")
    log(f"train_edges={train_graph.edge_index.shape[1]} val_edges={val_graph.edge_index.shape[1]}")
    teacher = spilu(
        A_scaled,
        drop_tol=args.spilu_drop_tol,
        fill_factor=args.spilu_fill_factor)
    log(f"teacher=spilu drop_tol={args.spilu_drop_tol} fill_factor={args.spilu_fill_factor}")

    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_gen = torch.Generator(device=device)
    train_gen.manual_seed(args.seed + 1)
    val_gen = torch.Generator(device=device)
    val_gen.manual_seed(args.seed + 2)
    val_probes = rademacher(
        num_nodes, args.val_probes, device, train_graph.x.dtype, val_gen)
    val_target = teacher_solve(teacher, val_probes)

    best_val = float("inf")
    best_epoch = -1
    for epoch in range(args.epochs):
        model.train()
        probes = rademacher(
            num_nodes, args.train_probes, device, train_graph.x.dtype, train_gen)
        target = teacher_solve(teacher, probes)
        pred, G_L_ei, G_L_val, G_U_ei, G_U_val = forward_action(
            model, train_graph, probes, device)
        imitation = relative_huber(
            pred, target,
            huber_beta=args.huber_beta,
            residual_clip=args.residual_clip)
        reg = value_regularization(G_L_ei, G_L_val, G_U_ei, G_U_val)
        loss = imitation + args.reg_weight * reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        did_validate = (epoch == 0) or ((epoch + 1) % args.val_freq == 0)
        did_solver_validate = (
            args.checkpoint_metric == "solver"
            and ((epoch == 0) or ((epoch + 1) % args.solver_val_freq == 0))
        )
        if did_validate:
            model.eval()
            with torch.no_grad():
                val_pred, vL_ei, vL_val, vU_ei, vU_val = forward_action(
                    model, val_graph, val_probes, device)
                val_imitation = relative_huber(
                    val_pred, val_target,
                    huber_beta=args.huber_beta,
                    residual_clip=args.residual_clip)
                val_reg = value_regularization(vL_ei, vL_val, vU_ei, vU_val)
                val_loss = val_imitation + args.reg_weight * val_reg
            solver_result = None
            if did_solver_validate:
                solver_result = solver_validation(
                    model, val_graph, A_scaled.tocsr(), device,
                    max_iter=args.solver_val_max_iter,
                    rtol=args.solver_val_rtol)
            checkpoint_value = (
                solver_result["metric"]
                if args.checkpoint_metric == "solver" and solver_result is not None
                else float(val_loss.item())
            )
            if checkpoint_value < best_val:
                best_val = checkpoint_value
                best_epoch = epoch
                torch.save(model.state_dict(), model_dir / "best_val.pt")
                log(f"saved best_val.pt epoch={epoch} metric={best_val:.6e}")
            solver_text = ""
            if solver_result is not None:
                solver_text = (
                    f" solver_metric={solver_result['metric']:.6e}"
                    f" solver_info={solver_result['info']}"
                    f" solver_iters={solver_result['iterations']}"
                    f" solver_min_rel={solver_result['min_rel']:.6e}")
            log(
                f"Epoch {epoch:04d} train={loss.item():.6e} "
                f"imit={imitation.item():.6e} reg={reg.item():.6e} "
                f"val={val_loss.item():.6e} val_imit={val_imitation.item():.6e} "
                f"val_reg={val_reg.item():.6e}{solver_text} best_epoch={best_epoch}")
        elif epoch % args.log_freq == 0:
            log(
                f"Epoch {epoch:04d} train={loss.item():.6e} "
                f"imit={imitation.item():.6e} reg={reg.item():.6e} best_epoch={best_epoch}")

    torch.save(model.state_dict(), model_dir / "latest_model.pt")
    meta = {
        "args": vars(args),
        "run_dir": str(run_dir),
        "best_val": best_val,
        "best_epoch": best_epoch,
        "matrix_scale": scale,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log(json.dumps({"best_val": best_val, "best_epoch": best_epoch}, indent=2))


if __name__ == "__main__":
    main()
