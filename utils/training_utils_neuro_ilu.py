"""Training utilities for Neuro-ILU: unsupervised Frobenius loss + BiCGSTAB validation."""

import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_sparse import spspmm
from torch_geometric.utils import to_dense_adj

sys.path.append('../')
from pcg import bicgstab_torch, bicgstab_sparse, build_neuro_ilu_csr, edge_to_csr


# ---------------------------------------------------------------------------
# Scipy ILU(0) + BiCGSTAB baseline
# ---------------------------------------------------------------------------

def _scipy_ilu0_bicgstab(A_torch, b_torch, tol=1e-8, max_iter=2000):
    """Run scipy ILU(0) factorization + scipy BiCGSTAB as a baseline.

    Returns iteration count, or max_iter if factorization/solve fails.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spilu, bicgstab, LinearOperator

    A_np = A_torch.cpu().numpy()
    b_np = b_torch.cpu().numpy().ravel()
    N = A_np.shape[0]

    try:
        A_sparse = csr_matrix(A_np)

        # ILU(0): no value-based dropping, no fill-in
        ilu = spilu(A_sparse, drop_tol=0.0, fill_factor=1.0)

        def precond_apply(r):
            return ilu.solve(r)

        M_op = LinearOperator((N, N), matvec=precond_apply, dtype=np.float64)

        iter_count = [0]

        def callback(xk):
            iter_count[0] += 1

        x0 = np.zeros(N)
        x, info = bicgstab(A_sparse, b_np, x0, tol=tol, maxiter=max_iter,
                           M=M_op, callback=callback, atol=0.0)

        if info == 0:
            return iter_count[0]
        else:
            return max_iter
    except Exception:
        return max_iter


# ---------------------------------------------------------------------------
# Unsupervised Frobenius loss
# ---------------------------------------------------------------------------


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


def _coalesce_entries(edge_index, values, N):
    order = torch.argsort(edge_index[0] * N + edge_index[1])
    edge_index = edge_index[:, order]
    values = values[order]

    flat = edge_index[0] * N + edge_index[1]
    unique_flat, inverse = torch.unique_consecutive(flat, return_inverse=True)
    coalesced = values.new_zeros((unique_flat.shape[0],))
    coalesced.index_add_(0, inverse, values)
    rows = torch.div(unique_flat, N, rounding_mode='floor')
    cols = torch.remainder(unique_flat, N)
    return torch.stack([rows, cols], dim=0), coalesced, unique_flat


def _align_sparse_values(source_edge_index, source_values, target_edge_index, N):
    source_edge_index, source_values, source_flat = _coalesce_entries(
        source_edge_index, source_values, N)
    target_flat = target_edge_index[0] * N + target_edge_index[1]
    pos = torch.searchsorted(source_flat, target_flat)
    valid = pos < source_flat.numel()

    aligned = source_values.new_zeros((target_flat.shape[0],))
    if valid.any():
        matched_pos = pos[valid]
        matched_mask = source_flat[matched_pos] == target_flat[valid]
        if matched_mask.any():
            valid_idx = torch.nonzero(valid, as_tuple=False).reshape(-1)
            aligned_idx = valid_idx[matched_mask]
            aligned[aligned_idx] = source_values[matched_pos[matched_mask]]
    return aligned


def sparse_pattern_mse_loss(L_edge_index, L_values, U_edge_index, U_values,
                            A_edge_index, A_values, N,
                            huber_beta=None, residual_clip=None):
    M_edge_index, M_values = spspmm(
        L_edge_index, L_values,
        U_edge_index, U_values,
        N, N, N)
    M_on_pattern = _align_sparse_values(M_edge_index, M_values, A_edge_index, N)
    if huber_beta is not None:
        denom = A_values.pow(2).mean().sqrt().clamp_min(1e-6)
        residual = (M_on_pattern - A_values) / denom
        if residual_clip is not None and residual_clip > 0:
            residual = residual.clamp(min=-residual_clip, max=residual_clip)
        return F.smooth_l1_loss(
            residual, torch.zeros_like(residual),
            beta=huber_beta, reduction='mean')
    return F.mse_loss(M_on_pattern, A_values)


def frobenius_loss(L_edge_index, L_values, U_edge_index, U_values,
                   A_edge_index, A_values, N, dirichlet_mask=None,
                   max_entries=None, deterministic=False,
                   huber_beta=None, residual_clip=None):
    """|| L @ U - A ||_F^2 computed only on the sparsity pattern of A.

    For each non-zero position (i,j) in A, we compare M_ij with A_ij.
    This enforces the ILU(0) fill-in constraint without needing ILU labels.

    Args:
        L_edge_index, L_values: sparse lower-triangular factor
        U_edge_index, U_values: sparse upper-triangular factor
        A_edge_index, A_values: sparse original matrix entries
        N: number of nodes
        dirichlet_mask: optional [N] bool mask of Dirichlet boundary nodes
        max_entries: optional cap on sampled A-pattern entries for scalability
        deterministic: use deterministic subsampling (validation) instead of random sampling

    Returns:
        scalar loss
    """
    A_edge_index, A_values = _subsample_pattern(
        A_edge_index, A_values, max_entries=max_entries, deterministic=deterministic)

    if dirichlet_mask is not None:
        active_mask = ~(dirichlet_mask[A_edge_index[0]] | dirichlet_mask[A_edge_index[1]])
        A_edge_index = A_edge_index[:, active_mask]
        A_values = A_values[active_mask]

    if A_values.numel() == 0:
        loss = L_values.new_zeros(())
    else:
        loss = sparse_pattern_mse_loss(
            L_edge_index, L_values, U_edge_index, U_values,
            A_edge_index, A_values, N,
            huber_beta=huber_beta,
            residual_clip=residual_clip)

    # Regularisation: penalise extreme values in L and U
    reg_L = (L_values ** 2).mean() * 1e-6
    reg_U = (U_values ** 2).mean() * 1e-6

    return loss + reg_L + reg_U


def _spmv(edge_index, values, x, N):
    row = edge_index[0]
    col = edge_index[1]
    if x.dim() == 1:
        out = x.new_zeros((N,))
        out.index_add_(0, row, values * x[col])
        return out

    out = x.new_zeros((N, x.shape[1]))
    out.index_add_(0, row, values.unsqueeze(-1) * x[col])
    return out


def _extract_u_diag(U_edge_index, U_values, N):
    diag_mask = U_edge_index[0] == U_edge_index[1]
    diag = U_values.new_zeros((N,))
    if diag_mask.any():
        diag_nodes = U_edge_index[0][diag_mask]
        diag[diag_nodes] = U_values[diag_mask]
    return diag


def operator_consistency_loss(L_edge_index, L_values, U_edge_index, U_values,
                              A_edge_index, A_values, rhs, true_x, N,
                              dirichlet_mask=None, random_probes=1,
                              huber_beta=1.0, residual_clip=10.0):
    """Measure how well LU matches A as a linear operator on true/random probes."""
    probes = [true_x.reshape(N, -1), rhs.reshape(N, -1)]
    for _ in range(max(0, random_probes)):
        probes.append(torch.randn((N, 1), device=true_x.device, dtype=true_x.dtype))

    total = L_values.new_zeros(())
    count = 0
    for probe in probes:
        Au = _spmv(A_edge_index, A_values, probe, N)
        Uu = _spmv(U_edge_index, U_values, probe, N)
        LUu = _spmv(L_edge_index, L_values, Uu, N)

        if dirichlet_mask is not None:
            active = ~dirichlet_mask
            diff = LUu[active] - Au[active]
            base = Au[active]
        else:
            diff = LUu - Au
            base = Au

        denom = base.pow(2).mean().sqrt().clamp_min(1e-6)
        rel_diff = diff / denom
        if residual_clip is not None and residual_clip > 0:
            rel_diff = rel_diff.clamp(min=-residual_clip, max=residual_clip)
        total = total + F.smooth_l1_loss(
            rel_diff, torch.zeros_like(rel_diff),
            beta=huber_beta, reduction='mean')
        count += 1

    return total / max(1, count)


def diagonal_barrier_loss(U_edge_index, U_values, A_edge_index, A_values, N,
                          floor_rel=0.1, floor_abs=1e-3):
    """Penalise U diagonals that fall below a target floor."""
    u_diag = _extract_u_diag(U_edge_index, U_values, N)

    a_diag = U_values.new_zeros((N,))
    a_diag_mask = A_edge_index[0] == A_edge_index[1]
    if a_diag_mask.any():
        a_diag_nodes = A_edge_index[0][a_diag_mask]
        a_diag[a_diag_nodes] = A_values[a_diag_mask]

    target = torch.maximum(a_diag.abs() * floor_rel, u_diag.new_full((N,), floor_abs))
    shortfall = F.relu(target - u_diag.abs())
    return (shortfall / target.clamp_min(1e-12)).pow(2).mean()


def pivot_regularization_loss(U_edge_index, U_values, threshold=1e-3, eps=1e-8):
    """Penalise dangerously small U pivots with an asymptotic barrier."""
    diag_mask = U_edge_index[0] == U_edge_index[1]
    if not diag_mask.any():
        return U_values.new_zeros(())
    u_diag = U_values[diag_mask]
    danger_mask = u_diag.abs() < threshold
    if not danger_mask.any():
        return U_values.new_zeros(())
    return torch.mean(1.0 / (u_diag[danger_mask].abs() + eps))


def implicit_inverse_loss(L_edge_index, L_values, U_edge_index, U_values,
                          A_edge_index, A_values, N, device,
                          dirichlet_mask=None, num_probes=1, max_nodes=-1,
                          pivot_threshold=1e-3, huber_beta=1.0,
                          residual_clip=10.0):
    """Robust proxy for ||(LU)^{-1} A - I|| on random Rademacher probes."""
    if num_probes <= 0:
        return L_values.new_zeros(())
    if max_nodes is not None and max_nodes > 0 and N > max_nodes:
        return L_values.new_zeros(())

    L_dense = to_dense_adj(L_edge_index, edge_attr=L_values, max_num_nodes=N).squeeze(0)
    U_dense = to_dense_adj(U_edge_index, edge_attr=U_values, max_num_nodes=N).squeeze(0)
    A_dense = to_dense_adj(A_edge_index, edge_attr=A_values, max_num_nodes=N).squeeze(0)
    if not (torch.isfinite(L_dense).all() and torch.isfinite(U_dense).all()
            and torch.isfinite(A_dense).all()):
        return L_values.new_zeros(())

    if dirichlet_mask is not None and dirichlet_mask.any():
        active = (~dirichlet_mask).to(L_dense.dtype)
        row_mask = active.unsqueeze(1)
        col_mask = active.unsqueeze(0)
        mask_2d = row_mask * col_mask
        A_dense = A_dense * mask_2d
        L_dense = L_dense * mask_2d
        U_dense = U_dense * mask_2d
        diag_idx = torch.arange(N, device=device)
        A_dense[diag_idx, diag_idx] = torch.where(
            dirichlet_mask,
            torch.ones(N, device=device, dtype=A_dense.dtype),
            A_dense[diag_idx, diag_idx])
        L_dense[diag_idx, diag_idx] = 1.0
        U_dense[diag_idx, diag_idx] = torch.where(
            dirichlet_mask,
            torch.ones(N, device=device, dtype=U_dense.dtype),
            U_dense[diag_idx, diag_idx])

    u_diag = torch.diagonal(U_dense)
    if dirichlet_mask is not None and dirichlet_mask.any():
        u_diag = u_diag[~dirichlet_mask]
    if (u_diag.numel() == 0 or not torch.isfinite(u_diag).all()
            or torch.any(u_diag.abs() < pivot_threshold)):
        return L_values.new_zeros(())

    total = L_values.new_zeros(())
    for _ in range(num_probes):
        v = torch.randint(0, 2, (N, 1), device=device, dtype=torch.int64)
        v = (v * 2 - 1).to(dtype=A_values.dtype)
        Av = A_dense @ v
        if not torch.isfinite(Av).all():
            return L_values.new_zeros(())
        w = torch.linalg.solve_triangular(L_dense, Av, upper=False, unitriangular=True)
        if not torch.isfinite(w).all():
            return L_values.new_zeros(())
        y = torch.linalg.solve_triangular(U_dense, w, upper=True, unitriangular=False)
        if not torch.isfinite(y).all():
            return L_values.new_zeros(())
        if dirichlet_mask is not None and dirichlet_mask.any():
            active = ~dirichlet_mask
            residual = y[active] - v[active]
            target = v[active]
        else:
            residual = y - v
            target = v

        denom = target.pow(2).mean().sqrt().clamp_min(1e-6)
        rel_residual = residual / denom
        if residual_clip is not None and residual_clip > 0:
            rel_residual = rel_residual.clamp(min=-residual_clip, max=residual_clip)
        total = total + F.smooth_l1_loss(
            rel_residual, torch.zeros_like(rel_residual),
            beta=huber_beta, reduction='mean')
    loss = total / num_probes
    if not torch.isfinite(loss):
        return L_values.new_zeros(())
    return loss


def compute_losses(args, model, node_attr, edge_attr, edge_index, u_next, rhs, batch_idx, data, device,
                   validation=False):
    pred_x, pred_rhs, (L_ei, L_val), (U_ei, U_val), (A_ei, A_val) = model(
        node_attr, edge_attr, edge_index,
        diag=data.diag if hasattr(data, 'diag') else None,
        input_r=data.r if hasattr(data, 'r') else None,
        input_x=torch.zeros_like(u_next, device=device),
        batch_idx=batch_idx,
        include_r=args.use_r if hasattr(args, 'use_r') else False,
        use_global=args.use_global if hasattr(args, 'use_global') else False,
        diagonalize=args.diagonalize if hasattr(args, 'diagonalize') else False,
        use_pred_x=args.use_pred_x if hasattr(args, 'use_pred_x') else False,
    )

    N = node_attr.shape[0]
    dirichlet_mask = node_attr[:, model.dirichlet_idx].to(torch.bool)
    frob_max_entries = args.frob_val_max_entries if validation else args.frob_train_max_entries

    frob = frobenius_loss(L_ei, L_val, U_ei, U_val, A_ei, A_val, N,
                          dirichlet_mask=dirichlet_mask,
                          max_entries=frob_max_entries,
                          deterministic=validation,
                          huber_beta=args.frob_loss_huber_beta,
                          residual_clip=args.frob_loss_residual_clip)
    op = operator_consistency_loss(
        L_ei, L_val, U_ei, U_val,
        A_ei, A_val,
        rhs=rhs,
        true_x=u_next,
        N=N,
        dirichlet_mask=dirichlet_mask,
        random_probes=args.operator_random_probes,
        huber_beta=args.operator_loss_huber_beta,
        residual_clip=args.operator_loss_residual_clip)
    diag_barrier = diagonal_barrier_loss(
        U_ei, U_val, A_ei, A_val, N,
        floor_rel=args.diag_floor_rel,
        floor_abs=args.diag_floor_abs)
    pivot_reg = pivot_regularization_loss(
        U_ei, U_val,
        threshold=args.pivot_reg_threshold,
        eps=args.pivot_reg_eps)
    inverse = implicit_inverse_loss(
        L_ei, L_val, U_ei, U_val,
        A_ei, A_val, N, device,
        dirichlet_mask=dirichlet_mask,
        num_probes=args.inverse_loss_probes,
        max_nodes=args.inverse_loss_max_nodes,
        pivot_threshold=args.inverse_loss_pivot_threshold,
        huber_beta=args.inverse_loss_huber_beta,
        residual_clip=args.inverse_loss_residual_clip)

    total = frob * args.frob_loss_weight
    total = total + op * args.operator_loss_weight
    total = total + diag_barrier * args.diag_barrier_weight
    total = total + pivot_reg * args.pivot_reg_weight
    total = total + inverse * args.inverse_loss_weight

    x_loss = None
    if args.x_loss_weight > 1e-6:
        x_loss = F.mse_loss(pred_x, u_next, reduction='mean')
        total = total + x_loss * args.x_loss_weight

    losses = {
        "total": total,
        "frob": frob,
        "operator": op,
        "diag_barrier": diag_barrier,
        "pivot_reg": pivot_reg,
        "inverse": inverse,
        "x": x_loss,
        "pred_x": pred_x,
        "pred_rhs": pred_rhs,
        "L_ei": L_ei,
        "L_val": L_val,
        "U_ei": U_ei,
        "U_val": U_val,
        "A_ei": A_ei,
        "A_val": A_val,
        "dirichlet_mask": dirichlet_mask,
    }
    return losses


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_epoch(args, train_loader, model, optimizer, lr_scheduler, device,
                tb_writer=None, tb_rate=1, total_steps=0, epoch=0, logger=None):
    """One training epoch with operator-aware Neuro-ILU losses."""

    total_loss = 0
    total_num = 0
    total_frob_loss = 0
    total_op_loss = 0
    total_diag_barrier_loss = 0
    total_pivot_reg_loss = 0
    total_inverse_loss = 0
    total_x_loss = 0

    model.train()
    tqdm_itr = tqdm(train_loader, position=1, desc='Training', leave=True)

    for i, data in enumerate(tqdm_itr):
        data = data.to(device)
        optimizer.zero_grad()

        node_attr = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        u_next = data.u_next
        batch_idx = data.batch
        losses = compute_losses(
            args, model, node_attr, edge_attr, edge_index,
            u_next, data.rhs, batch_idx, data, device, validation=False)
        loss = losses["total"]
        frob_loss = losses["frob"]
        op_loss = losses["operator"]
        diag_barrier_loss_val = losses["diag_barrier"]
        pivot_reg_loss_val = losses["pivot_reg"]
        inverse_loss_val = losses["inverse"]

        if not torch.isfinite(loss):
            if logger is not None:
                logger.info(f'Iter {i} non-finite loss, skipping batch')
            continue

        total_frob_loss += frob_loss.item()
        total_op_loss += op_loss.item()
        total_diag_barrier_loss += diag_barrier_loss_val.item()
        total_pivot_reg_loss += pivot_reg_loss_val.item()
        total_inverse_loss += inverse_loss_val.item()
        total_num += 1

        x_loss_val = 0.0
        if losses["x"] is not None:
            x_loss_val = losses["x"].item()
            total_x_loss += x_loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if not 'ReduceLROnPlateau' in str(lr_scheduler.__class__):
            lr_scheduler.step()

        total_loss += loss.item()
        total_steps += 1

        log_str = (f'Train loss: {loss.item():.4e} | '
                   f'frob: {frob_loss.item():.4e} | '
                   f'op: {op_loss.item():.4e} | '
                   f'diag: {diag_barrier_loss_val.item():.4e} | '
                   f'pivot: {pivot_reg_loss_val.item():.4e} | '
                   f'inv: {inverse_loss_val.item():.4e}')
        if x_loss_val > 0:
            log_str += f' | x: {x_loss_val:.4e}'
        tqdm_itr.set_postfix_str(log_str)

        if logger is not None and (tb_rate <= 1 or i % tb_rate == 0 or i == len(train_loader) - 1):
            logger.info(f'Iter {i} ' + log_str)

    avg_loss = total_loss / max(1, total_num)
    avg_frob = total_frob_loss / max(1, total_num)
    avg_op = total_op_loss / max(1, total_num)
    avg_diag = total_diag_barrier_loss / max(1, total_num)
    avg_pivot = total_pivot_reg_loss / max(1, total_num)
    avg_inverse = total_inverse_loss / max(1, total_num)

    if tb_writer is not None:
        tb_writer.add_scalar("Loss_Train/total", avg_loss, global_step=total_steps)
        tb_writer.add_scalar("Loss_Train/frobenius", avg_frob, global_step=total_steps)
        tb_writer.add_scalar("Loss_Train/operator", avg_op, global_step=total_steps)
        tb_writer.add_scalar("Loss_Train/diag_barrier", avg_diag, global_step=total_steps)
        tb_writer.add_scalar("Loss_Train/pivot_reg", avg_pivot, global_step=total_steps)
        tb_writer.add_scalar("Loss_Train/inverse", avg_inverse, global_step=total_steps)

    if logger is not None:
        logger.info(
            f'Train epoch summary: total={avg_loss:.4e} | frob={avg_frob:.4e} | '
            f'op={avg_op:.4e} | diag={avg_diag:.4e} | pivot={avg_pivot:.4e} | '
            f'inv={avg_inverse:.4e} | batches={total_num}')

    return avg_loss, total_steps


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

def val_epoch(args, val_loader, model, device,
              tb_writer=None, total_steps=0, epoch=0,
              save_dir='results/', logger=None):
    """Validation: compute operator-aware losses + run BiCGSTAB comparison."""

    total_frob_loss = 0
    total_op_loss = 0
    total_diag_barrier_loss = 0
    total_pivot_reg_loss = 0
    total_inverse_loss = 0
    total_num = 0
    total_neuro_iter = 0
    total_ilu0_iter = 0
    total_jacobi_iter = 0
    total_identity_iter = 0
    num_shapes = 0
    num_neuro_fail = 0
    num_ilu0_fail = 0
    num_jacobi_fail = 0
    num_ident_fail = 0

    model.eval()
    tqdm_itr = tqdm(val_loader, position=1, desc='Validation', leave=True)

    for i, data in enumerate(tqdm_itr):
        data = data.to(device)

        node_attr = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        u_next = data.u_next
        rhs = data.rhs
        batch_idx = data.batch
        with torch.no_grad():
            losses = compute_losses(
                args, model, node_attr, edge_attr, edge_index,
                u_next, rhs, batch_idx, data, device, validation=True)

        total_frob_loss += losses["frob"].item()
        total_op_loss += losses["operator"].item()
        total_diag_barrier_loss += losses["diag_barrier"].item()
        total_pivot_reg_loss += losses["pivot_reg"].item()
        total_inverse_loss += losses["inverse"].item()
        total_num += 1
        num_shapes += 1

        # Run BiCGSTAB simulation with learned preconditioner and baselines
        if args.simulate:
            max_it = 2000
            neuro_iter, ilu0_iter, jacobi_iter, ident_iter = neuro_ilu_simulate(
                args, model, data, losses["L_ei"], losses["L_val"], losses["U_ei"], losses["U_val"],
                losses["A_ei"], losses["A_val"], node_attr.shape[0], device, epoch, save_dir)
            total_neuro_iter += neuro_iter
            total_ilu0_iter += ilu0_iter
            total_jacobi_iter += jacobi_iter
            total_identity_iter += ident_iter
            if neuro_iter >= max_it: num_neuro_fail += 1
            if ilu0_iter >= max_it: num_ilu0_fail += 1
            if jacobi_iter >= max_it: num_jacobi_fail += 1
            if ident_iter >= max_it: num_ident_fail += 1

            tqdm_itr.set_postfix_str(
                f'Val frob: {losses["frob"].item():.4e} | '
                f'op: {losses["operator"].item():.4e} | '
                f'diag: {losses["diag_barrier"].item():.4e} | '
                f'pivot: {losses["pivot_reg"].item():.4e} | '
                f'inv: {losses["inverse"].item():.4e} | '
                f'Neuro: {neuro_iter} | ILU0: {ilu0_iter} | '
                f'Jac: {jacobi_iter} | Id: {ident_iter}')

    avg_frob = total_frob_loss / max(1, total_num)
    avg_op = total_op_loss / max(1, total_num)
    avg_diag = total_diag_barrier_loss / max(1, total_num)
    avg_pivot = total_pivot_reg_loss / max(1, total_num)
    avg_inverse = total_inverse_loss / max(1, total_num)
    avg_total = (
        avg_frob * args.frob_loss_weight
        + avg_op * args.operator_loss_weight
        + avg_diag * args.diag_barrier_weight
        + avg_pivot * args.pivot_reg_weight
        + avg_inverse * args.inverse_loss_weight
    )
    avg_neuro_iter = total_neuro_iter / max(1, num_shapes) if num_shapes > 0 else 0
    avg_ilu0_iter = total_ilu0_iter / max(1, num_shapes) if num_shapes > 0 else 0
    avg_jacobi_iter = total_jacobi_iter / max(1, num_shapes) if num_shapes > 0 else 0
    avg_ident_iter = total_identity_iter / max(1, num_shapes) if num_shapes > 0 else 0

    if tb_writer is not None:
        tb_writer.add_scalar("Loss_Val/total", avg_total, global_step=total_steps)
        tb_writer.add_scalar("Loss_Val/frobenius", avg_frob, global_step=total_steps)
        tb_writer.add_scalar("Loss_Val/operator", avg_op, global_step=total_steps)
        tb_writer.add_scalar("Loss_Val/diag_barrier", avg_diag, global_step=total_steps)
        tb_writer.add_scalar("Loss_Val/pivot_reg", avg_pivot, global_step=total_steps)
        tb_writer.add_scalar("Loss_Val/inverse", avg_inverse, global_step=total_steps)
        tb_writer.add_scalar("Solver/neuro_ilu_iters", avg_neuro_iter, global_step=total_steps)
        tb_writer.add_scalar("Solver/ilu0_iters", avg_ilu0_iter, global_step=total_steps)
        tb_writer.add_scalar("Solver/jacobi_iters", avg_jacobi_iter, global_step=total_steps)
        tb_writer.add_scalar("Solver/identity_iters", avg_ident_iter, global_step=total_steps)

    if logger is not None and num_shapes > 0:
        logger.info(
            f'Val summary: total={avg_total:.4e} | frob={avg_frob:.4e} | op={avg_op:.4e} | '
            f'diag={avg_diag:.4e} | pivot={avg_pivot:.4e} | inv={avg_inverse:.4e} | '
            f'Neuro: {avg_neuro_iter:.1f}it (fail:{num_neuro_fail}) | '
            f'ILU0: {avg_ilu0_iter:.1f}it (fail:{num_ilu0_fail}) | '
            f'Jacobi: {avg_jacobi_iter:.1f}it (fail:{num_jacobi_fail}) | '
            f'Identity: {avg_ident_iter:.1f}it (fail:{num_ident_fail})')

    return avg_neuro_iter, avg_total


# ---------------------------------------------------------------------------
# BiCGSTAB simulation harness
# ---------------------------------------------------------------------------

def neuro_ilu_simulate(args, model, data, L_ei, L_val, U_ei, U_val,
                       A_ei, A_val, N, device, epoch, save_dir):
    """Run BiCGSTAB with Neuro-ILU (sparse), ILU(0) via scipy, Jacobi, and Identity.

    Neuro-ILU uses sparse triangular solves to avoid O(N^2) densification.
    All four methods use BiCGSTAB so the comparison isolates preconditioner quality.
    """

    # Dirichlet mask (SuiteSparse has none, but keep for generality)
    dirichlet_mask = data.x[:, model.dirichlet_idx].to(torch.bool)
    has_dirichlet = dirichlet_mask.any().item()

    # --- Build A and b ---
    A_csr = edge_to_csr(A_ei, A_val, N)
    b_np = data.rhs.cpu().numpy().ravel()

    if has_dirichlet:
        # Apply Dirichlet BC to A and b (modify CSR in-place is complex;
        # fall back to dense for correctness when boundaries exist)
        from torch_geometric.utils import to_dense_adj
        A_dense = to_dense_adj(A_ei, edge_attr=A_val, max_num_nodes=N).squeeze(0)
        A_dense = A_dense.clone().cpu().numpy()
        b_np = data.rhs.cpu().numpy().ravel().copy()
        dns = torch.where(dirichlet_mask)[0].cpu().numpy()
        for dn in dns:
            A_dense[dn, :] = 0
            A_dense[:, dn] = 0
            A_dense[dn, dn] = 1.0
            b_np[dn] = 0.0
        from scipy.sparse import csr_matrix
        A_csr = csr_matrix(A_dense)

    options = {'abs_tol': 1e-9, 'rel_tol': 1e-8, 'max_iter': 2000}

    # --- 1. Neuro-ILU + sparse BiCGSTAB ---
    L_csr, U_csr = build_neuro_ilu_csr(
        L_ei, L_val, U_ei, U_val, N, dirichlet_mask=dirichlet_mask if has_dirichlet else None)

    neuro_iter, _ = bicgstab_sparse(A_csr, b_np, L_csr, U_csr,
                                     tol=options['rel_tol'], max_iter=options['max_iter'])

    # --- 2. ILU(0) via scipy + scipy BiCGSTAB ---
    # Need a dense A on torch for _scipy_ilu0_bicgstab
    from torch_geometric.utils import to_dense_adj
    A_dense_t = to_dense_adj(A_ei, edge_attr=A_val, max_num_nodes=N).squeeze(0)
    b_t = data.rhs.reshape(-1, 1)
    if has_dirichlet:
        A_dense_t = A_dense_t.clone()
        b_t = b_t.clone()
        for dn in torch.where(dirichlet_mask)[0]:
            A_dense_t[dn, :] = 0
            A_dense_t[:, dn] = 0
            A_dense_t[dn, dn] = 1.0
            b_t[dn] = 0.0
    ilu0_iter = _scipy_ilu0_bicgstab(A_dense_t, b_t, tol=1e-8, max_iter=options['max_iter'])

    # --- 3. Jacobi + torch BiCGSTAB (dense, L=I U=diag(A)) ---
    A_dense = A_dense_t.clone().double()
    b_mod = b_t.clone().double().to(device)
    diag_A = torch.diag(A_dense).to(device)
    diag_A = torch.where(diag_A == 0, torch.ones_like(diag_A), diag_A)
    jacobi_L = torch.eye(N, device=device, dtype=torch.float64)
    jacobi_U = torch.diag(diag_A).to(device=device, dtype=torch.float64)
    jacobi_iter, _ = bicgstab_torch(A_dense, b_mod, jacobi_L, jacobi_U, options, device=device)

    # --- 4. Identity + torch BiCGSTAB ---
    ident_L = torch.eye(N, device=device, dtype=torch.float64)
    ident_U = torch.eye(N, device=device, dtype=torch.float64)
    ident_iter, _ = bicgstab_torch(A_dense, b_mod, ident_L, ident_U, options, device=device)

    print(f'[N={N}] Neuro-ILU: {neuro_iter} | ILU(0): {ilu0_iter} | '
          f'Jacobi: {jacobi_iter} | Identity: {ident_iter}')

    return neuro_iter, ilu0_iter, jacobi_iter, ident_iter
