import argparse
import pathlib
import sys

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model_neuro_ilu import Net
from pcg import bicgstab_sparse, edge_to_csr, build_neuro_ilu_csr, bicgstab_torch


def load_first_sample(data_path):
    data = np.load(data_path, allow_pickle=True)
    sample = data[0]
    return {
        "x": torch.as_tensor(sample["x"]).float(),
        "edge_attr": torch.as_tensor(sample["edge_attr"]).float(),
        "edge_index": torch.as_tensor(sample["edge_index"]).long(),
        "rhs": torch.as_tensor(sample["rhs"]).float(),
        "diag": torch.as_tensor(sample["diag"]).reshape(-1, 1).float(),
        "r": torch.as_tensor(sample["r"]).float(),
        "u_next": torch.as_tensor(sample["u_next"]).float(),
    }


def build_args():
    args = argparse.Namespace()
    args.dataset = "heatmultisource"
    args.hidden_dim = 8
    args.hidden_layers_encoder = 1
    args.hidden_layers_decoder = 1
    args.hidden_layers_processor = 1
    args.num_iterations = 2
    args.norm = "LayerNorm"
    args.use_global = False
    args.use_r = False
    args.diagonalize = False
    args.use_pred_x = True
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default="dataset/diffusivity_100.0/circle_low_res/data_0000.npy",
        help="Path to a local heat dataset .npy shard",
    )
    parsed = parser.parse_args()

    sample = load_first_sample(parsed.data_path)
    args = build_args()

    model = Net(
        args,
        in_dim_node=sample["x"].shape[-1],
        in_dim_edge=sample["edge_attr"].shape[-1],
        out_dim=1,
        b_dim=sample["x"].shape[0],
        num_edges=sample["edge_attr"].shape[0],
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
    model.eval()

    with torch.no_grad():
        _, _, (L_ei, L_val), (U_ei, U_val), (A_ei, A_val) = model(
            sample["x"],
            sample["edge_attr"],
            sample["edge_index"],
            diag=sample["diag"],
            input_r=sample["r"],
            input_x=torch.zeros_like(sample["u_next"]),
            batch_idx=torch.zeros(sample["x"].shape[0], dtype=torch.long),
            include_r=False,
            use_global=False,
            diagonalize=False,
            use_pred_x=True,
        )

    N = sample["x"].shape[0]
    dirichlet_mask = sample["x"][:, 3].to(torch.bool)
    A_csr = edge_to_csr(A_ei, A_val, N)
    b_np = sample["rhs"].cpu().numpy().ravel()
    L_csr, U_csr = build_neuro_ilu_csr(L_ei, L_val, U_ei, U_val, N, dirichlet_mask=dirichlet_mask)
    options = {"abs_tol": 1e-9, "rel_tol": 1e-8, "max_iter": 2000}

    neuro_iter, _ = bicgstab_sparse(A_csr, b_np, L_csr, U_csr, tol=options["rel_tol"], max_iter=options["max_iter"])

    from scipy.sparse.linalg import spilu, bicgstab, LinearOperator

    iter_count = [0]

    def callback(_xk):
        iter_count[0] += 1

    ilu = spilu(A_csr, drop_tol=0.0, fill_factor=1.0)
    ilu_op = LinearOperator(A_csr.shape, matvec=ilu.solve, dtype=np.float64)
    _, info = bicgstab(A_csr, b_np, x0=np.zeros(N), tol=1e-8, maxiter=options["max_iter"], M=ilu_op, callback=callback, atol=0.0)
    ilu_iter = iter_count[0] if info == 0 else options["max_iter"]

    A_dense = torch.tensor(A_csr.toarray(), dtype=torch.float64)
    b_t = torch.tensor(b_np.reshape(-1, 1), dtype=torch.float64)
    diag_A = torch.diag(A_dense)
    diag_A = torch.where(diag_A == 0, torch.ones_like(diag_A), diag_A)
    jacobi_iter, _ = bicgstab_torch(
        A_dense, b_t,
        torch.eye(N, dtype=torch.float64),
        torch.diag(diag_A),
        options,
        device="cpu",
    )
    ident_iter, _ = bicgstab_torch(
        A_dense, b_t,
        torch.eye(N, dtype=torch.float64),
        torch.eye(N, dtype=torch.float64),
        options,
        device="cpu",
    )

    print(f"data_path={parsed.data_path}")
    print(f"N={N}, nnz={A_csr.nnz}")
    print(f"Neuro-ILU iterations: {neuro_iter}")
    print(f"ILU(0) iterations:    {ilu_iter}")
    print(f"Jacobi iterations:    {jacobi_iter}")
    print(f"Identity iterations:  {ident_iter}")


if __name__ == "__main__":
    main()
