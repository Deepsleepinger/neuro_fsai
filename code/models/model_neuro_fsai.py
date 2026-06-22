"""Neuro-FSAI: directed sparse approximate inverse factors.

The model predicts two fixed sparse factors, G_L and G_U, and the solver applies
the preconditioner as y = G_U @ (G_L @ r).  This avoids triangular solves in the
Krylov loop; preconditioning is two sparse matvecs.
"""

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, sort_edge_index

from .network_module import MLP, MP


class PDEDirectedConv(MessagePassing):
    """Directed PDE message passing without attention softmax normalization."""

    def __init__(self, node_dim, edge_dim, hidden_dim=None):
        super().__init__(aggr='add')
        hidden_dim = hidden_dim or node_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.layer_norm(
            self.propagate(edge_index, x=x, edge_attr=edge_attr))

    def message(self, x_i, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))


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
        super().__init__()
        self.args = args
        self.dirichlet_idx = dirichlet_idx

        if 'heat' in args.dataset:
            self.pde = 'heat'
        elif 'flow' in args.dataset:
            self.pde = 'flow'
        elif 'wave' in args.dataset:
            self.pde = 'wave'
        else:
            self.pde = 'syn'

        self.node_encoder = MLP(
            in_dim_node, out_dim_node,
            hidden_dim_node, hidden_layers_node, norm_type)
        self.edge_encoder = MLP(
            in_dim_edge, out_dim_edge,
            hidden_dim_edge, hidden_layers_edge, norm_type)

        self.mp_layers = nn.ModuleList()
        for _ in range(num_iterations):
            self.mp_layers.append(PDEDirectedConv(
                node_dim=out_dim_node,
                edge_dim=out_dim_edge,
                hidden_dim=hidden_dim_processor_node))
        self.processor_activation = nn.ReLU()

        edge_decoder_dim = 2 * out_dim_node + out_dim_edge
        self.edge_decoder = MLP(
            edge_decoder_dim, 1,
            hidden_dim_decoder, hidden_layers_decoder, norm_type=None)
        self._zero_init_edge_decoder()
        self.node_decoder_x = MLP(
            hidden_dim_processor_node, out_dim,
            hidden_dim_decoder, hidden_layers_decoder, norm_type=None)
        self.spmv = MP()

    def _zero_init_edge_decoder(self):
        """Start from the Jacobi preconditioner: learned off-diagonals are zero."""
        for layer in reversed(self.edge_decoder.model):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                return

    def _extract_A_entries(self, edge_attr):
        if self.pde == 'heat':
            return edge_attr[:, -1] + edge_attr[:, -2]
        if self.pde == 'flow':
            return edge_attr[:, -1]
        if self.pde == 'wave':
            return edge_attr[:, 1]
        return edge_attr[:, -1]

    def _extract_diag(self, edge_index, A_values, num_nodes, diag=None):
        if diag is not None:
            return diag.reshape(-1).to(device=A_values.device, dtype=A_values.dtype)

        source = edge_index[0]
        target = edge_index[1]
        diag_mask = source == target
        diag = A_values.new_zeros((num_nodes,))
        if diag_mask.any():
            diag[source[diag_mask]] = A_values[diag_mask]
        return diag

    def _safe_inv_sqrt_abs_diag(self, diag_values):
        eps = getattr(self.args, 'fsai_jacobi_eps', 1e-12)
        abs_diag = diag_values.abs()
        return torch.where(
            abs_diag > eps,
            torch.rsqrt(abs_diag.clamp_min(eps)),
            torch.ones_like(abs_diag))

    def _diag_bases(self, edge_index, A_diag):
        nodes = edge_index[0]
        a_diag = A_diag[nodes]
        sign = torch.where(a_diag < 0, -torch.ones_like(a_diag), torch.ones_like(a_diag))
        inv_sqrt_abs = self._safe_inv_sqrt_abs_diag(a_diag)

        gl_diag = inv_sqrt_abs
        gu_diag = sign * inv_sqrt_abs
        return gl_diag, gu_diag

    def _relative_edge_attr(self, edge_attr, edge_index, A_values, A_diag):
        source = edge_index[0]
        target = edge_index[1]
        diag_mask = source == target
        effective_values = A_values.clone()
        if diag_mask.any():
            effective_values[diag_mask] = A_diag[source[diag_mask]]
        inv_scale = self._safe_inv_sqrt_abs_diag(A_diag[source])
        inv_scale = inv_scale * self._safe_inv_sqrt_abs_diag(A_diag[target])
        relative_values = effective_values * inv_scale

        clip = getattr(self.args, 'fsai_relative_value_clip', 10.0)
        if clip is not None and clip > 0:
            relative_values = relative_values.clamp(min=-clip, max=clip)

        encoded_attr = edge_attr.clone()
        encoded_attr[:, -1] = relative_values
        return encoded_attr

    def forward(self, node_attr, edge_attr, edge_index,
                diag=None, input_r=None, input_x=None, batch_idx=None,
                include_r=False, use_global=False, diagonalize=False,
                use_pred_x=False):
        num_nodes = node_attr.shape[0]
        device = node_attr.device

        edge_index, edge_attr = add_remaining_self_loops(
            edge_index, edge_attr=edge_attr, fill_value=0.0, num_nodes=num_nodes)

        edge_index, edge_attr = sort_edge_index(edge_index, edge_attr=edge_attr)
        A_values = self._extract_A_entries(edge_attr)
        A_diag = self._extract_diag(edge_index, A_values, num_nodes, diag=diag)
        source = edge_index[0]
        target = edge_index[1]
        diag_mask = source == target
        if diag_mask.any():
            A_values = A_values.clone()
            A_values[diag_mask] = A_diag[source[diag_mask]]
        encoder_edge_attr = self._relative_edge_attr(edge_attr, edge_index, A_values, A_diag)

        node_feat = self.node_encoder(node_attr)
        edge_feat = self.edge_encoder(encoder_edge_attr)

        x = node_feat
        for mp_l in self.mp_layers:
            x = self.processor_activation(
                mp_l(x, edge_index, edge_attr=edge_feat))

        decoded_x = torch.clamp(self.node_decoder_x(x), -10.0, 10.0)

        # v3.0: keep directed edge identity visible at decode time.  For
        # reciprocal edges, [x_src, x_dst, e] and [x_dst, x_src, e] are distinct.
        decoder_edge_feat = torch.cat([x[source], x[target], edge_feat], dim=-1)
        raw_edge_values = torch.clamp(self.edge_decoder(decoder_edge_feat).squeeze(-1), -20.0, 20.0)

        # edge_index is PyG directed source->target.  For a matrix entry A[row,col],
        # source=col and target=row, so lower/upper tests use target/source.
        lower_strict = target > source
        upper_strict = target < source

        offdiag_scale = getattr(self.args, 'fsai_offdiag_scale', 0.1)
        offdiag_basis = self._safe_inv_sqrt_abs_diag(A_diag[source])
        offdiag_basis = offdiag_basis * self._safe_inv_sqrt_abs_diag(A_diag[target])
        basis_cap = getattr(self.args, 'fsai_offdiag_basis_cap', 1.0)
        if basis_cap is not None and basis_cap > 0:
            offdiag_basis = offdiag_basis.clamp(max=basis_cap)
        offdiag_values = offdiag_scale * torch.tanh(raw_edge_values) * offdiag_basis

        G_L_edge_index = edge_index[:, lower_strict]
        G_L_values = offdiag_values[lower_strict]
        G_U_edge_index = edge_index[:, upper_strict]
        G_U_values = offdiag_values[upper_strict]

        diag_nodes = torch.arange(num_nodes, device=device)
        diag_edge_index = torch.stack([diag_nodes, diag_nodes], dim=0)
        gl_diag, gu_diag = self._diag_bases(diag_edge_index, A_diag)

        G_L_edge_index = torch.cat([G_L_edge_index, diag_edge_index], dim=1)
        G_L_values = torch.cat([G_L_values, gl_diag], dim=0)
        G_U_edge_index = torch.cat([G_U_edge_index, diag_edge_index], dim=1)
        G_U_values = torch.cat([G_U_values, gu_diag], dim=0)

        if input_r is None:
            input_r = torch.zeros((num_nodes, 1), device=device, dtype=node_attr.dtype)
        precond_r = self.spmv(input_r, G_L_values.unsqueeze(-1), G_L_edge_index)
        precond_r = self.spmv(precond_r, G_U_values.unsqueeze(-1), G_U_edge_index)

        if not use_pred_x:
            output_x = input_x if input_x is not None else torch.zeros_like(decoded_x)
        else:
            output_x = decoded_x

        return (output_x, precond_r,
                (G_L_edge_index, G_L_values),
                (G_U_edge_index, G_U_values),
                (edge_index, A_values))
