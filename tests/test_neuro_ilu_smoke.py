import pathlib
import sys
import types
import unittest

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

    def to_dense_adj(edge_index, edge_attr=None, max_num_nodes=None):
        if torch is None:
            raise RuntimeError("torch is required for the torch_geometric stub")
        num_nodes = int(max_num_nodes if max_num_nodes is not None else edge_index.max().item() + 1)
        dense = torch.zeros((1, num_nodes, num_nodes), dtype=edge_attr.dtype, device=edge_attr.device)
        dense[0, edge_index[0], edge_index[1]] = edge_attr
        return dense

    class _DummyDataLoader:
        pass

    tg_utils.to_dense_adj = to_dense_adj
    tg_data.DataLoader = _DummyDataLoader
    tg.data = tg_data
    tg.utils = tg_utils
    sys.modules.setdefault("torch_geometric", tg)
    sys.modules.setdefault("torch_geometric.utils", tg_utils)
    sys.modules.setdefault("torch_geometric.data", tg_data)


if torch is not None:
    _install_torch_geometric_stub()
    from pcg import apply_neuro_ilu_preconditioner, build_neuro_ilu_csr
    from utils.training_utils_neuro_ilu import (
        diagonal_barrier_loss,
        frobenius_loss,
        implicit_inverse_loss,
        operator_consistency_loss,
        pivot_regularization_loss,
        sparse_pattern_mse_loss,
    )


@unittest.skipIf(torch is None, "torch is not installed in this environment; smoke test skipped")
class NeuroILUSmokeTest(unittest.TestCase):
    def test_sparse_frobenius_loss_zero_for_exact_factorization(self):
        L_edge_index = torch.tensor([[0, 1, 1, 2, 2], [0, 0, 1, 1, 2]], dtype=torch.long)
        L_values = torch.tensor([1.0, 2.0, 1.0, 3.0, 1.0], dtype=torch.float64)
        U_edge_index = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 1, 2, 2]], dtype=torch.long)
        U_values = torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)

        A_dense = np.array([[4.0, 5.0, 0.0],
                            [8.0, 16.0, 7.0],
                            [0.0, 18.0, 29.0]])
        A_edge_index = torch.from_numpy(np.vstack(np.nonzero(A_dense))).long()
        A_values = torch.tensor(A_dense[A_dense != 0], dtype=torch.float64)

        loss = frobenius_loss(
            L_edge_index, L_values, U_edge_index, U_values,
            A_edge_index, A_values, N=3, dirichlet_mask=None)

        expected_reg = 1e-6 * ((L_values ** 2).mean() + (U_values ** 2).mean())
        self.assertAlmostEqual(loss.item(), expected_reg.item(), places=10)

    def test_sparse_pattern_mse_loss_zero_for_exact_factorization(self):
        L_edge_index = torch.tensor([[0, 1, 1, 2, 2], [0, 0, 1, 1, 2]], dtype=torch.long)
        L_values = torch.tensor([1.0, 2.0, 1.0, 3.0, 1.0], dtype=torch.float64)
        U_edge_index = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 1, 2, 2]], dtype=torch.long)
        U_values = torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)

        A_dense = np.array([[4.0, 5.0, 0.0],
                            [8.0, 16.0, 7.0],
                            [0.0, 18.0, 29.0]])
        A_edge_index = torch.from_numpy(np.vstack(np.nonzero(A_dense))).long()
        A_values = torch.tensor(A_dense[A_dense != 0], dtype=torch.float64)

        loss = sparse_pattern_mse_loss(
            L_edge_index, L_values, U_edge_index, U_values,
            A_edge_index, A_values, N=3)
        self.assertAlmostEqual(loss.item(), 0.0, places=10)

    def test_sampled_sparse_frobenius_loss_zero_for_exact_factorization(self):
        L_edge_index = torch.tensor([[0, 1, 1, 2, 2], [0, 0, 1, 1, 2]], dtype=torch.long)
        L_values = torch.tensor([1.0, 2.0, 1.0, 3.0, 1.0], dtype=torch.float64)
        U_edge_index = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 1, 2, 2]], dtype=torch.long)
        U_values = torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)

        A_dense = np.array([[4.0, 5.0, 0.0],
                            [8.0, 16.0, 7.0],
                            [0.0, 18.0, 29.0]])
        A_edge_index = torch.from_numpy(np.vstack(np.nonzero(A_dense))).long()
        A_values = torch.tensor(A_dense[A_dense != 0], dtype=torch.float64)

        loss = frobenius_loss(
            L_edge_index, L_values, U_edge_index, U_values,
            A_edge_index, A_values, N=3, dirichlet_mask=None,
            max_entries=2, deterministic=True)

        expected_reg = 1e-6 * ((L_values ** 2).mean() + (U_values ** 2).mean())
        self.assertAlmostEqual(loss.item(), expected_reg.item(), places=10)

    def test_sparse_preconditioner_matches_numpy_triangular_solves(self):
        L_edge_index = torch.tensor([[0, 1, 1, 2, 2], [0, 0, 1, 1, 2]], dtype=torch.long)
        L_values = torch.tensor([1.0, 2.0, 1.0, 3.0, 1.0], dtype=torch.float64)
        U_edge_index = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 1, 2, 2]], dtype=torch.long)
        U_values = torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
        rhs = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)

        L_csr, U_csr = build_neuro_ilu_csr(
            L_edge_index, L_values, U_edge_index, U_values, N=3, dirichlet_mask=None)
        y = apply_neuro_ilu_preconditioner(
            rhs, L_edge_index, L_values, U_edge_index, U_values, N=3)

        L_dense = L_csr.toarray()
        U_dense = U_csr.toarray()
        expected = np.linalg.solve(U_dense, np.linalg.solve(L_dense, rhs.numpy().reshape(-1)))

        np.testing.assert_allclose(y.detach().cpu().numpy().reshape(-1), expected, rtol=1e-10, atol=1e-10)

    def test_operator_and_diag_losses_are_finite(self):
        L_edge_index = torch.tensor([[0, 1, 1, 2, 2], [0, 0, 1, 1, 2]], dtype=torch.long)
        L_values = torch.tensor([1.0, 2.0, 1.0, 3.0, 1.0], dtype=torch.float64)
        U_edge_index = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 1, 2, 2]], dtype=torch.long)
        U_values = torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)

        A_dense = np.array([[4.0, 5.0, 0.0],
                            [8.0, 16.0, 7.0],
                            [0.0, 18.0, 29.0]])
        A_edge_index = torch.from_numpy(np.vstack(np.nonzero(A_dense))).long()
        A_values = torch.tensor(A_dense[A_dense != 0], dtype=torch.float64)
        rhs = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        true_x = torch.tensor([[0.5], [-1.0], [2.0]], dtype=torch.float64)

        op_loss = operator_consistency_loss(
            L_edge_index, L_values, U_edge_index, U_values,
            A_edge_index, A_values,
            rhs=rhs, true_x=true_x, N=3, dirichlet_mask=None, random_probes=0)
        diag_loss = diagonal_barrier_loss(
            U_edge_index, U_values, A_edge_index, A_values,
            N=3, floor_rel=0.1, floor_abs=1e-3)
        pivot_loss = pivot_regularization_loss(
            U_edge_index, U_values, threshold=10.0, eps=1e-6)
        inverse_loss = implicit_inverse_loss(
            L_edge_index, L_values, U_edge_index, U_values,
            A_edge_index, A_values, N=3,
            device=torch.device("cpu"),
            dirichlet_mask=None, num_probes=2, max_nodes=16,
            pivot_threshold=1e-3)

        self.assertTrue(torch.isfinite(op_loss))
        self.assertTrue(torch.isfinite(diag_loss))
        self.assertTrue(torch.isfinite(pivot_loss))
        self.assertTrue(torch.isfinite(inverse_loss))
        self.assertAlmostEqual(op_loss.item(), 0.0, places=10)
        self.assertAlmostEqual(diag_loss.item(), 0.0, places=10)
        expected_pivot = torch.mean(1.0 / (torch.tensor([4.0, 6.0, 8.0], dtype=torch.float64) + 1e-6))
        self.assertAlmostEqual(pivot_loss.item(), expected_pivot.item(), places=10)
        self.assertAlmostEqual(inverse_loss.item(), 0.0, places=10)

    def test_inverse_loss_skips_dangerous_pivots(self):
        L_edge_index = torch.tensor([[0, 1, 1], [0, 0, 1]], dtype=torch.long)
        L_values = torch.tensor([1.0, 0.25, 1.0], dtype=torch.float64)
        U_edge_index = torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.long)
        U_values = torch.tensor([1e-8, 2.0, 1.0], dtype=torch.float64)

        A_dense = np.array([[1e-8, 2.0],
                            [2.5e-9, 1.5]])
        A_edge_index = torch.from_numpy(np.vstack(np.nonzero(A_dense))).long()
        A_values = torch.tensor(A_dense[A_dense != 0], dtype=torch.float64)

        inverse_loss = implicit_inverse_loss(
            L_edge_index, L_values, U_edge_index, U_values,
            A_edge_index, A_values, N=2,
            device=torch.device("cpu"),
            dirichlet_mask=None, num_probes=2, max_nodes=16,
            pivot_threshold=1e-3)

        self.assertTrue(torch.isfinite(inverse_loss))
        self.assertAlmostEqual(inverse_loss.item(), 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
