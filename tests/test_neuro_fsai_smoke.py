import pathlib
import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def _install_torch_geometric_stub():
    tg = types.ModuleType("torch_geometric")
    tg_utils = types.ModuleType("torch_geometric.utils")
    tg_data = types.ModuleType("torch_geometric.data")

    def add_remaining_self_loops(edge_index, edge_attr=None, fill_value=0.0, num_nodes=None):
        if num_nodes is None:
            num_nodes = int(edge_index.max().item()) + 1
        existing = set(zip(edge_index[0].cpu().tolist(), edge_index[1].cpu().tolist()))
        add_nodes = [i for i in range(num_nodes) if (i, i) not in existing]
        if not add_nodes:
            return edge_index, edge_attr
        loops = torch.tensor([add_nodes, add_nodes], dtype=edge_index.dtype, device=edge_index.device)
        edge_index = torch.cat([edge_index, loops], dim=1)
        if edge_attr is not None:
            fill = torch.full(
                (len(add_nodes), edge_attr.shape[1]),
                float(fill_value), dtype=edge_attr.dtype, device=edge_attr.device)
            edge_attr = torch.cat([edge_attr, fill], dim=0)
        return edge_index, edge_attr

    def sort_edge_index(edge_index, edge_attr=None):
        n = int(edge_index.max().item()) + 1
        order = torch.argsort(edge_index[0] * n + edge_index[1])
        if edge_attr is None:
            return edge_index[:, order]
        return edge_index[:, order], edge_attr[order]

    class _DummyDataLoader:
        pass

    tg_utils.add_remaining_self_loops = add_remaining_self_loops
    tg_utils.sort_edge_index = sort_edge_index
    tg_data.DataLoader = _DummyDataLoader
    tg.data = tg_data
    tg.utils = tg_utils
    sys.modules.setdefault("torch_geometric", tg)
    sys.modules.setdefault("torch_geometric.utils", tg_utils)
    sys.modules.setdefault("torch_geometric.data", tg_data)


if torch is not None:
    try:
        import torch_geometric  # noqa: F401
    except ModuleNotFoundError:
        _install_torch_geometric_stub()
    import torch.nn as nn
    from models.model_neuro_fsai import Net
    from pcg import apply_neuro_fsai_preconditioner, build_neuro_fsai_csr
    from utils.training_utils_neuro_fsai import (
        fsai_inverse_probe_loss,
        fsai_rhs_loss,
        fsai_value_regularization,
    )
    from scipy.sparse import csr_matrix
    from utils.convert_suitesparse import matrix_to_graph
    from utils.topology_expansion import expand_topology_to_2hop_scipy


@unittest.skipIf(torch is None, "torch is not installed in this environment; smoke test skipped")
class NeuroFSAISmokeTest(unittest.TestCase):
    def test_processor_uses_additive_directed_pde_conv(self):
        args = SimpleNamespace(dataset="suitesparse")
        model = Net(
            args,
            in_dim_node=4,
            in_dim_edge=3,
            out_dim=1,
            b_dim=1,
            out_dim_node=4,
            out_dim_edge=4,
            hidden_dim_node=4,
            hidden_dim_edge=4,
            hidden_layers_node=1,
            hidden_layers_edge=1,
            num_iterations=2,
            hidden_dim_processor_node=4,
            hidden_dim_processor_edge=4,
            hidden_layers_processor_node=1,
            hidden_layers_processor_edge=1,
            hidden_dim_decoder=4,
            hidden_layers_decoder=1,
            dirichlet_idx=3,
        )

        self.assertEqual([layer.__class__.__name__ for layer in model.mp_layers],
                         ["PDEDirectedConv", "PDEDirectedConv"])
        self.assertTrue(all(layer.aggr == "add" for layer in model.mp_layers))
        self.assertTrue(all(isinstance(layer.layer_norm, nn.LayerNorm)
                            for layer in model.mp_layers))
        self.assertTrue(all(layer.layer_norm.normalized_shape == (4,)
                            for layer in model.mp_layers))

    def test_model_sorts_edge_features_before_encoding(self):
        class LastFeatureDecoder(nn.Module):
            def forward(self, edge_feat):
                return edge_feat[:, -1:]

        args = SimpleNamespace(
            dataset="suitesparse",
            fsai_offdiag_scale=1.0,
            fsai_diag_scale=0.0,
            fsai_diag_abs_floor=0.01,
        )
        model = Net(
            args,
            in_dim_node=4,
            in_dim_edge=3,
            out_dim=1,
            b_dim=1,
            out_dim_node=3,
            out_dim_edge=3,
            hidden_dim_node=3,
            hidden_dim_edge=3,
            hidden_layers_node=1,
            hidden_layers_edge=1,
            num_iterations=0,
            hidden_dim_processor_node=3,
            hidden_dim_processor_edge=3,
            hidden_layers_processor_node=1,
            hidden_layers_processor_edge=1,
            hidden_dim_decoder=3,
            hidden_layers_decoder=1,
            dirichlet_idx=3,
        ).double()
        model.edge_encoder = nn.Identity()
        model.edge_decoder = LastFeatureDecoder()

        node_attr = torch.zeros((3, 4), dtype=torch.float64)
        edge_index = torch.tensor(
            [[2, 0, 1, 0],
             [0, 2, 0, 1]],
            dtype=torch.long)
        raw_values = torch.tensor([0.7, 0.2, -0.4, 0.5], dtype=torch.float64)
        edge_attr = torch.stack([
            torch.zeros_like(raw_values),
            torch.zeros_like(raw_values),
            raw_values,
        ], dim=-1)

        _, _, (G_L_ei, G_L_val), (G_U_ei, G_U_val), _ = model(
            node_attr, edge_attr, edge_index,
            input_r=torch.zeros((3, 1), dtype=torch.float64),
            input_x=torch.zeros((3, 1), dtype=torch.float64))

        lower = {
            tuple(G_L_ei[:, i].tolist()): G_L_val[i].item()
            for i in range(G_L_ei.shape[1])
            if G_L_ei[0, i] != G_L_ei[1, i]
        }
        upper = {
            tuple(G_U_ei[:, i].tolist()): G_U_val[i].item()
            for i in range(G_U_ei.shape[1])
            if G_U_ei[0, i] != G_U_ei[1, i]
        }

        self.assertAlmostEqual(lower[(0, 1)], torch.tanh(torch.tensor(0.5)).item(), places=7)
        self.assertAlmostEqual(lower[(0, 2)], torch.tanh(torch.tensor(0.2)).item(), places=7)
        self.assertAlmostEqual(upper[(1, 0)], torch.tanh(torch.tensor(-0.4)).item(), places=7)
        self.assertAlmostEqual(upper[(2, 0)], torch.tanh(torch.tensor(0.7)).item(), places=7)

    def test_edge_decoder_sees_directed_source_target_order(self):
        class SourceMinusTargetDecoder(nn.Module):
            def forward(self, edge_feat):
                node_dim = 4
                return (edge_feat[:, :1] - edge_feat[:, node_dim:node_dim + 1])

        args = SimpleNamespace(
            dataset="suitesparse",
            fsai_offdiag_scale=1.0,
            fsai_diag_scale=0.0,
            fsai_diag_abs_floor=0.01,
            fsai_jacobi_eps=1e-12,
            fsai_relative_value_clip=10.0,
            fsai_offdiag_basis_cap=1.0,
        )
        model = Net(
            args,
            in_dim_node=4,
            in_dim_edge=3,
            out_dim=1,
            b_dim=1,
            out_dim_node=4,
            out_dim_edge=3,
            hidden_dim_node=4,
            hidden_dim_edge=3,
            hidden_layers_node=1,
            hidden_layers_edge=1,
            num_iterations=0,
            hidden_dim_processor_node=4,
            hidden_dim_processor_edge=3,
            hidden_layers_processor_node=1,
            hidden_layers_processor_edge=1,
            hidden_dim_decoder=4,
            hidden_layers_decoder=1,
            dirichlet_idx=3,
        ).double()
        model.node_encoder = nn.Identity()
        model.edge_encoder = nn.Identity()
        model.edge_decoder = SourceMinusTargetDecoder()

        node_attr = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0],
             [-1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float64)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_attr = torch.tensor(
            [[0.0, 0.0, 1.0],
             [0.0, 0.0, 1.0]],
            dtype=torch.float64)
        diag = torch.ones((2, 1), dtype=torch.float64)

        _, _, (G_L_ei, G_L_val), (G_U_ei, G_U_val), _ = model(
            node_attr, edge_attr, edge_index,
            diag=diag,
            input_r=torch.zeros((2, 1), dtype=torch.float64),
            input_x=torch.zeros((2, 1), dtype=torch.float64))

        lower = {
            tuple(G_L_ei[:, i].tolist()): G_L_val[i].item()
            for i in range(G_L_ei.shape[1])
            if G_L_ei[0, i] != G_L_ei[1, i]
        }
        upper = {
            tuple(G_U_ei[:, i].tolist()): G_U_val[i].item()
            for i in range(G_U_ei.shape[1])
            if G_U_ei[0, i] != G_U_ei[1, i]
        }

        self.assertAlmostEqual(lower[(0, 1)], torch.tanh(torch.tensor(2.0)).item(), places=7)
        self.assertAlmostEqual(upper[(1, 0)], torch.tanh(torch.tensor(-2.0)).item(), places=7)

    def test_fsai_losses_zero_for_exact_inverse(self):
        A = torch.tensor([[4.0, 1.0], [2.0, 3.0]], dtype=torch.float64)
        A_inv = torch.linalg.inv(A)
        A_edge_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=torch.long)
        A_values = torch.tensor([4.0, 1.0, 2.0, 3.0], dtype=torch.float64)

        G_L_edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        G_L_values = torch.ones(2, dtype=torch.float64)
        G_U_edge_index = torch.from_numpy(np.vstack(np.nonzero(A_inv.numpy()))).long().flip(0)
        G_U_values = torch.tensor(A_inv.numpy()[np.nonzero(A_inv.numpy())], dtype=torch.float64)

        torch.manual_seed(0)
        inv_loss = fsai_inverse_probe_loss(
            G_L_edge_index, G_L_values,
            G_U_edge_index, G_U_values,
            A_edge_index, A_values,
            N=2, device=torch.device("cpu"),
            dirichlet_mask=None, num_probes=8)
        rhs = torch.tensor([[1.0], [-2.0]], dtype=torch.float64)
        true_x = A_inv @ rhs
        rhs_loss = fsai_rhs_loss(
            G_L_edge_index, G_L_values,
            G_U_edge_index, G_U_values,
            rhs=rhs, true_x=true_x, N=2)

        self.assertAlmostEqual(inv_loss.item(), 0.0, places=10)
        self.assertAlmostEqual(rhs_loss.item(), 0.0, places=10)

    def test_zero_residual_model_is_jacobi_preconditioner(self):
        args = SimpleNamespace(
            dataset="suitesparse",
            fsai_offdiag_scale=1.0,
            fsai_diag_scale=0.0,
            fsai_diag_abs_floor=0.01,
            fsai_jacobi_eps=1e-12,
            fsai_relative_value_clip=10.0,
            fsai_offdiag_basis_cap=1.0,
        )
        model = Net(
            args,
            in_dim_node=4,
            in_dim_edge=3,
            out_dim=1,
            b_dim=1,
            out_dim_node=4,
            out_dim_edge=4,
            hidden_dim_node=4,
            hidden_dim_edge=4,
            hidden_layers_node=1,
            hidden_layers_edge=1,
            num_iterations=0,
            hidden_dim_processor_node=4,
            hidden_dim_processor_edge=4,
            hidden_layers_processor_node=1,
            hidden_layers_processor_edge=1,
            hidden_dim_decoder=4,
            hidden_layers_decoder=1,
            dirichlet_idx=3,
        ).double()

        node_attr = torch.zeros((2, 4), dtype=torch.float64)
        edge_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=torch.long)
        values = torch.tensor([4.0, 1.0, 2.0, 9.0], dtype=torch.float64)
        edge_attr = torch.stack([
            torch.zeros_like(values),
            (edge_index[0] == edge_index[1]).to(values.dtype),
            values,
        ], dim=-1)
        diag = torch.tensor([[4.0], [9.0]], dtype=torch.float64)
        rhs = torch.tensor([[8.0], [18.0]], dtype=torch.float64)

        _, precond_rhs, (_, G_L_val), (_, G_U_val), _ = model(
            node_attr, edge_attr, edge_index,
            diag=diag,
            input_r=rhs,
            input_x=torch.zeros_like(rhs))

        expected = rhs / diag
        np.testing.assert_allclose(
            precond_rhs.detach().cpu().numpy(),
            expected.numpy(), rtol=1e-12, atol=1e-12)
        self.assertGreater(G_L_val.numel(), 0)
        self.assertGreater(G_U_val.numel(), 0)

    def test_forward_uses_passed_diag_for_added_self_loops(self):
        args = SimpleNamespace(
            dataset="suitesparse",
            fsai_offdiag_scale=0.0,
            fsai_diag_scale=0.0,
            fsai_diag_abs_floor=0.01,
            fsai_jacobi_eps=1e-12,
            fsai_relative_value_clip=10.0,
            fsai_offdiag_basis_cap=1.0,
        )
        model = Net(
            args,
            in_dim_node=4,
            in_dim_edge=3,
            out_dim=1,
            b_dim=1,
            out_dim_node=4,
            out_dim_edge=4,
            hidden_dim_node=4,
            hidden_dim_edge=4,
            hidden_layers_node=1,
            hidden_layers_edge=1,
            num_iterations=0,
            hidden_dim_processor_node=4,
            hidden_dim_processor_edge=4,
            hidden_layers_processor_node=1,
            hidden_layers_processor_edge=1,
            hidden_dim_decoder=4,
            hidden_layers_decoder=1,
            dirichlet_idx=3,
        ).double()

        node_attr = torch.zeros((2, 4), dtype=torch.float64)
        edge_index = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)
        values = torch.tensor([1.0, 2.0], dtype=torch.float64)
        edge_attr = torch.stack([
            torch.zeros_like(values),
            torch.zeros_like(values),
            values,
        ], dim=-1)
        diag = torch.tensor([[4.0], [9.0]], dtype=torch.float64)

        _, _, _, _, (A_ei, A_val) = model(
            node_attr, edge_attr, edge_index,
            diag=diag,
            input_r=torch.zeros((2, 1), dtype=torch.float64),
            input_x=torch.zeros((2, 1), dtype=torch.float64))

        actual_diag = {
            int(A_ei[0, i].item()): A_val[i].item()
            for i in range(A_ei.shape[1])
            if A_ei[0, i] == A_ei[1, i]
        }
        self.assertEqual(actual_diag, {0: 4.0, 1: 9.0})

    def test_offdiag_basis_cap_prevents_small_pivot_amplification(self):
        class ConstantDecoder(nn.Module):
            def forward(self, edge_feat):
                return torch.full(
                    (edge_feat.shape[0], 1),
                    10.0,
                    dtype=edge_feat.dtype,
                    device=edge_feat.device)

        args = SimpleNamespace(
            dataset="suitesparse",
            fsai_offdiag_scale=0.1,
            fsai_diag_scale=0.0,
            fsai_diag_abs_floor=0.01,
            fsai_jacobi_eps=1e-12,
            fsai_relative_value_clip=10.0,
            fsai_offdiag_basis_cap=1.0,
        )
        model = Net(
            args,
            in_dim_node=4,
            in_dim_edge=3,
            out_dim=1,
            b_dim=1,
            out_dim_node=3,
            out_dim_edge=3,
            hidden_dim_node=3,
            hidden_dim_edge=3,
            hidden_layers_node=1,
            hidden_layers_edge=1,
            num_iterations=0,
            hidden_dim_processor_node=3,
            hidden_dim_processor_edge=3,
            hidden_layers_processor_node=1,
            hidden_layers_processor_edge=1,
            hidden_dim_decoder=3,
            hidden_layers_decoder=1,
            dirichlet_idx=3,
        ).double()
        model.edge_decoder = ConstantDecoder()

        node_attr = torch.zeros((2, 4), dtype=torch.float64)
        edge_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=torch.long)
        values = torch.tensor([1e-12, 1.0, 1.0, 1e-12], dtype=torch.float64)
        edge_attr = torch.stack([
            torch.zeros_like(values),
            (edge_index[0] == edge_index[1]).to(values.dtype),
            values,
        ], dim=-1)
        diag = torch.tensor([[1e-12], [1e-12]], dtype=torch.float64)

        _, _, (G_L_ei, G_L_val), (G_U_ei, G_U_val), _ = model(
            node_attr, edge_attr, edge_index,
            diag=diag,
            input_r=torch.zeros((2, 1), dtype=torch.float64),
            input_x=torch.zeros((2, 1), dtype=torch.float64))

        offdiag_l = G_L_val[G_L_ei[0] != G_L_ei[1]]
        offdiag_u = G_U_val[G_U_ei[0] != G_U_ei[1]]
        self.assertLessEqual(float(offdiag_l.abs().max()), 0.1)
        self.assertLessEqual(float(offdiag_u.abs().max()), 0.1)

    def test_sparse_fsai_preconditioner_matches_csr_matvec(self):
        G_L_edge_index = torch.tensor([[0, 1, 1], [0, 0, 1]], dtype=torch.long)
        G_L_values = torch.tensor([1.0, 0.5, 1.0], dtype=torch.float64)
        G_U_edge_index = torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.long)
        G_U_values = torch.tensor([2.0, -1.0, 3.0], dtype=torch.float64)
        rhs = torch.tensor([[1.0], [2.0]], dtype=torch.float64)

        G_L_csr, G_U_csr = build_neuro_fsai_csr(
            G_L_edge_index, G_L_values, G_U_edge_index, G_U_values, N=2)
        y = apply_neuro_fsai_preconditioner(
            rhs, G_L_edge_index, G_L_values, G_U_edge_index, G_U_values, N=2)
        expected = G_U_csr @ (G_L_csr @ rhs.numpy().reshape(-1))

        np.testing.assert_allclose(
            y.detach().cpu().numpy().reshape(-1),
            expected, rtol=1e-12, atol=1e-12)

    def test_fsai_regularization_ignores_jacobi_diagonal(self):
        diag_edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        large_diag_values = torch.tensor([1e4, -1e4], dtype=torch.float64)

        reg = fsai_value_regularization(
            diag_edge_index, large_diag_values,
            diag_edge_index, large_diag_values)

        self.assertEqual(reg.item(), 0.0)

    def test_matrix_to_graph_topology_hop2_adds_zero_valued_candidates(self):
        A = csr_matrix(
            np.array([
                [1.0, 2.0, 0.0],
                [0.0, 3.0, 4.0],
                [0.0, 0.0, 5.0],
            ]))

        graph_h1 = matrix_to_graph(A, name="chain", topology_hop=1)
        graph_h2 = matrix_to_graph(A, name="chain", topology_hop=2)

        self.assertGreater(graph_h2["edge_index"].shape[1], graph_h1["edge_index"].shape[1])
        self.assertEqual(graph_h2["meta"]["nnz"], A.nnz)
        self.assertEqual(graph_h2["meta"]["topology_hop"], 2)

        edges = {
            tuple(graph_h2["edge_index"][:, i].tolist()): graph_h2["edge_attr"][i, 1]
            for i in range(graph_h2["edge_index"].shape[1])
        }
        # A[0, 2] is structurally absent, but A^2 has path 2 -> 1 -> 0.
        self.assertIn((2, 0), edges)
        self.assertEqual(edges[(2, 0)], 0.0)

        # Original directed edge for A[0, 1] keeps the true matrix value.
        self.assertEqual(edges[(1, 0)], 2.0)

    def test_matrix_to_graph_topology_hop2_can_fallback_when_too_dense(self):
        A = csr_matrix(
            np.array([
                [1.0, 2.0, 0.0],
                [0.0, 3.0, 4.0],
                [0.0, 0.0, 5.0],
            ]))

        graph_h1 = matrix_to_graph(A, name="chain", topology_hop=1)
        graph_capped = matrix_to_graph(
            A, name="chain", topology_hop=2, max_topology_ratio=1.0)

        self.assertEqual(graph_capped["edge_index"].shape[1], graph_h1["edge_index"].shape[1])
        self.assertFalse(graph_capped["meta"]["topology_expanded"])

    def test_expand_topology_to_2hop_scipy_adds_zero_attr_edges(self):
        edge_index = torch.tensor(
            [[0, 1, 2],
             [1, 2, 2]],
            dtype=torch.long)
        edge_attr = torch.tensor(
            [[1.0, 0.0, 2.0],
             [1.0, 0.0, 3.0],
             [0.0, 1.0, 5.0]],
            dtype=torch.float32)

        expanded_index, expanded_attr, expanded = expand_topology_to_2hop_scipy(
            edge_index, edge_attr, num_nodes=3)

        self.assertTrue(expanded)
        edges = {
            tuple(expanded_index[:, i].tolist()): expanded_attr[i].tolist()
            for i in range(expanded_index.shape[1])
        }
        self.assertIn((0, 2), edges)
        self.assertEqual(edges[(0, 2)], [0.0, 0.0, 0.0])

    def test_expand_topology_cap_uses_merged_edge_count(self):
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2],
             [1, 0, 2, 0, 1]],
            dtype=torch.long)
        edge_attr = torch.ones((edge_index.shape[1], 3), dtype=torch.float32)

        expanded_index, _, expanded = expand_topology_to_2hop_scipy(
            edge_index, edge_attr, num_nodes=3, max_topology_ratio=2.0)

        self.assertTrue(expanded)
        self.assertLessEqual(expanded_index.shape[1], 2 * edge_index.shape[1])


if __name__ == "__main__":
    unittest.main()
