''' Dual-Channel Neuro-ILU: learns L and U factors for non-symmetric matrices.

    Instead of enforcing symmetry via to_undirected + LL^T, this model:
    1. Preserves directed edges from the FEM graph (= the sparsity pattern of A)
    2. Topologically splits predicted edge values into:
       - L: strict lower-triangular entries + hard-coded unit diagonal
       - U: upper-triangular entries including the diagonal (boosted by A's diagonal)

    The output (L_sparse, U_sparse) is a linear operator — safe to plug into BiCGSTAB/GMRES.
'''

import math
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import sort_edge_index, add_remaining_self_loops

from .network_module import MLP, MeshMP, MP


class Net(nn.Module):
    def __init__(self,
            args,
            in_dim_node,
            in_dim_edge,
            out_dim,
            b_dim,
            num_edges=31260,
            out_dim_node=128,
            out_dim_edge=128,
            hidden_dim_node=128,
            hidden_dim_edge=128,
            hidden_layers_node=2,
            hidden_layers_edge=2,
            num_iterations=30,
            hidden_dim_processor_node=128,
            hidden_dim_processor_edge=128,
            hidden_layers_processor_node=2,
            hidden_layers_processor_edge=2,
            norm_type='LayerNorm',
            hidden_dim_decoder=128,
            hidden_layers_decoder=2,
            dirichlet_idx=3,
            global_pool=False,
            **kwargs):
        super(Net, self).__init__()
        self.args = args

        if 'heat' in args.dataset:
            self.pde = 'heat'
        elif 'flow' in args.dataset:
            self.pde = 'flow'
        elif 'wave' in args.dataset:
            self.pde = 'wave'
        else:
            self.pde = 'syn'
        self.dirichlet_idx = dirichlet_idx

        # --- Encoders ---
        self.node_encoder = MLP(in_dim_node, out_dim_node,
            hidden_dim_node, hidden_layers_node, norm_type)
        self.edge_encoder = MLP(in_dim_edge, out_dim_edge,
            hidden_dim_edge, hidden_layers_edge, norm_type)

        # --- Processor (message passing) ---
        self.mp_layers = nn.ModuleList()
        for i in range(num_iterations):
            self.mp_layers.append(MeshMP(out_dim_node, out_dim_edge,
                out_dim_node, out_dim_edge,
                hidden_dim_processor_node, hidden_dim_processor_edge,
                hidden_layers_processor_node, hidden_layers_processor_edge, norm_type))

        # --- Decoders ---
        # edge_decoder outputs one scalar per directed edge
        self.edge_decoder = MLP(hidden_dim_processor_edge, 1,
                                hidden_dim_decoder, hidden_layers_decoder, norm_type=None)

        # Optional: node decoder for x prediction (auxiliary loss)
        self.node_decoder_x = MLP(hidden_dim_processor_node, out_dim,
                                  hidden_dim_decoder, hidden_layers_decoder, norm_type=None)

        # Non-parametric message-passing for sparse mat-vec
        self.spmv = MP()

    def _extract_diag_ele(self, edge_attr, edge_index):
        """Extract diagonal elements of A from edge features.

        Returns a [N, 1] tensor. Nodes without a self-loop edge get 0.
        """
        num_nodes = int(edge_index.max()) + 1
        if self.pde == 'heat':
            values = edge_attr[:, -1] + edge_attr[:, -2]
        elif self.pde == 'flow':
            values = edge_attr[:, -1]
        elif self.pde == 'wave':
            values = edge_attr[:, 1]
        else:
            values = edge_attr[:, -1]

        is_diag = (edge_index[0, :] == edge_index[1, :])
        diag_nodes = edge_index[0, is_diag]
        diag_vals = values[is_diag]

        diag_full = torch.zeros(num_nodes, 1, device=edge_attr.device, dtype=edge_attr.dtype)
        diag_full[diag_nodes, 0] = diag_vals
        return diag_full

    def _extract_A_entries(self, edge_attr):
        """Extract A values per edge (for unsupervised loss)."""
        if self.pde == 'heat':
            return edge_attr[:, -1] + edge_attr[:, -2]
        elif self.pde == 'flow':
            return edge_attr[:, -1]
        elif self.pde == 'wave':
            return edge_attr[:, 1]
        else:
            return edge_attr[:, -1]

    def forward(self, node_attr, edge_attr, edge_index,
                diag=None, input_r=None, input_x=None, batch_idx=None,
                include_r=False, use_global=False, diagonalize=False,
                use_pred_x=False):
        """
        Returns:
            pred_x_next:           predicted next state [N, out_dim]
            (L_edge_index, L_values), (U_edge_index, U_values): sparse LU factors
            A_edge_index, A_values:                         original matrix entries (for loss)
        """
        num_nodes = node_attr.shape[0]
        device = node_attr.device

        # Ensure every node has a self-loop (some sparse matrices have zero diagonals)
        edge_index, edge_attr = add_remaining_self_loops(
            edge_index, edge_attr=edge_attr, fill_value=0.0, num_nodes=num_nodes)

        # --- Encode ---
        node_feat = self.node_encoder(node_attr)
        edge_feat = self.edge_encoder(edge_attr)

        # Dirichlet mask
        dirichlet_mask = node_attr[:, self.dirichlet_idx].to(torch.bool)

        # Sort edges for deterministic processing
        edge_index, edge_attr = sort_edge_index(edge_index, edge_attr=edge_attr)

        # --- Message passing ---
        x = node_feat
        for mp_l in self.mp_layers:
            x, edge_feat = mp_l(x, edge_index, edge_feat)

        # --- Decode ---
        decoded_x = self.node_decoder_x(x)  # [N, out_dim]
        decoded_x = torch.clamp(decoded_x, -10.0, 10.0)
        pred_edge_vals = self.edge_decoder(edge_feat).squeeze(-1)  # [E]
        pred_edge_vals = torch.clamp(pred_edge_vals, -10.0, 10.0)

        # --- Topological split into L and U ---
        row, col = edge_index[0], edge_index[1]

        # U: upper triangular (row <= col) — includes diagonal
        mask_U = (row <= col)
        U_edge_index = edge_index[:, mask_U]
        U_values = pred_edge_vals[mask_U]

        # L: strict lower triangular (row > col)
        mask_L = (row > col)
        L_edge_index = edge_index[:, mask_L]
        L_values = pred_edge_vals[mask_L]

        # --- Enforce structural constraints ---
        # Extract ground-truth diagonal of A for numerical safety
        diag_ele = self._extract_diag_ele(edge_attr, edge_index)  # [N, 1]

        # U diagonal: preserve A's pivot sign and enforce a non-zero magnitude floor.
        is_U_diag = (U_edge_index[0] == U_edge_index[1])
        U_diag_nodes = U_edge_index[0][is_U_diag]  # which nodes have diagonal in U
        U_values = U_values.clone()
        a_diag = diag_ele[U_diag_nodes].squeeze(-1)
        diag_sign = torch.where(a_diag < 0, -torch.ones_like(a_diag), torch.ones_like(a_diag))
        diag_floor_rel = getattr(self.args, 'u_diag_floor_rel', 1e-3) if hasattr(self, 'args') else 1e-3
        diag_floor_abs = getattr(self.args, 'u_diag_floor_abs', 1e-3) if hasattr(self, 'args') else 1e-3
        diag_floor = torch.maximum(
            a_diag.abs() * diag_floor_rel,
            a_diag.new_full(a_diag.shape, diag_floor_abs))
        diag_delta = F.softplus(U_values[is_U_diag])
        U_values[is_U_diag] = a_diag + diag_sign * (diag_delta + diag_floor)

        # L diagonal: hard-code to 1.0 (unit lower-triangular)
        diag_nodes = torch.arange(num_nodes, device=device)
        L_diag_edge_index = torch.stack([diag_nodes, diag_nodes], dim=0)
        L_diag_values = torch.ones(num_nodes, device=device)

        # Concatenate L strict-lower with diagonal
        L_edge_index = torch.cat([L_edge_index, L_diag_edge_index], dim=1)
        L_values = torch.cat([L_values, L_diag_values], dim=0)

        # --- Extract A entries for loss computation ---
        A_values = self._extract_A_entries(edge_attr)  # per-edge values of A

        # --- Handle Dirichlet boundary: set L and U rows/cols to identity ---
        # This mirrors what the original code does with A[dirichlet_node] = 0, diag=1
        # For the preconditioner, boundary nodes should act as identity
        # We handle this during solver application, not in the model output

        # --- Optional: compute pred_rhs = L @ U @ x for auxiliary loss ---
        # Ux = U @ x
        Ux = self.spmv(decoded_x, U_values.unsqueeze(-1), U_edge_index)
        # LUx = L @ Ux
        LUx = self.spmv(Ux, L_values.unsqueeze(-1), L_edge_index)

        # Apply Dirichlet mask to predicted rhs
        LUx[dirichlet_mask] = decoded_x[dirichlet_mask]

        if not use_pred_x:
            output_x = input_x if input_x is not None else torch.zeros_like(decoded_x)
        else:
            output_x = decoded_x

        return (output_x, LUx,
                (L_edge_index, L_values),
                (U_edge_index, U_values),
                (edge_index, A_values))
