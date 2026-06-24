"""Training utilities for Neuro-FSAI.

The core objective is solver-facing: y = G_U G_L A v should approximate v.
This keeps training on sparse matvecs only and matches the BiCGSTAB
preconditioner application used at evaluation time.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _spmv(edge_index, values, x, N):
    source = edge_index[0]
    target = edge_index[1]
    if x.dim() == 1:
        out = x.new_zeros((N,))
        out.index_add_(0, target, values * x[source])
        return out

    out = x.new_zeros((N, x.shape[1]))
    out.index_add_(0, target, values.unsqueeze(-1) * x[source])
    return out


def _relative_huber(residual, target, huber_beta=1.0, residual_clip=10.0):
    denom = target.pow(2).mean().sqrt().clamp_min(1e-6)
    rel = residual / denom
    if residual_clip is not None and residual_clip > 0:
        rel = rel.clamp(min=-residual_clip, max=residual_clip)
    return F.smooth_l1_loss(
        rel, torch.zeros_like(rel),
        beta=huber_beta, reduction='mean')


def fsai_inverse_probe_loss(G_L_edge_index, G_L_values,
                            G_U_edge_index, G_U_values,
                            A_edge_index, A_values,
                            N, device,
                            dirichlet_mask=None,
                            num_probes=4,
                            huber_beta=1.0,
                            residual_clip=10.0):
    if num_probes <= 0:
        return G_L_values.new_zeros(())

    total = G_L_values.new_zeros(())
    for _ in range(num_probes):
        v = torch.randint(0, 2, (N, 1), device=device, dtype=torch.int64)
        v = (v * 2 - 1).to(dtype=A_values.dtype)
        Av = _spmv(A_edge_index, A_values, v, N)
        z = _spmv(G_L_edge_index, G_L_values, Av, N)
        y = _spmv(G_U_edge_index, G_U_values, z, N)

        if dirichlet_mask is not None and dirichlet_mask.any():
            active = ~dirichlet_mask
            residual = y[active] - v[active]
            target = v[active]
        else:
            residual = y - v
            target = v
        total = total + _relative_huber(
            residual, target,
            huber_beta=huber_beta,
            residual_clip=residual_clip)

    return total / num_probes


def fsai_rhs_loss(G_L_edge_index, G_L_values,
                  G_U_edge_index, G_U_values,
                  rhs, true_x, N,
                  dirichlet_mask=None,
                  huber_beta=1.0,
                  residual_clip=10.0):
    z = _spmv(G_L_edge_index, G_L_values, rhs.reshape(N, -1), N)
    y = _spmv(G_U_edge_index, G_U_values, z, N)
    true_x = true_x.reshape(N, -1)

    if dirichlet_mask is not None and dirichlet_mask.any():
        active = ~dirichlet_mask
        residual = y[active] - true_x[active]
        target = true_x[active]
    else:
        residual = y - true_x
        target = true_x
    return _relative_huber(
        residual, target,
        huber_beta=huber_beta,
        residual_clip=residual_clip)


def fsai_value_regularization(G_L_edge_index, G_L_values, G_U_edge_index, G_U_values):
    lower_offdiag = G_L_edge_index[0] != G_L_edge_index[1]
    upper_offdiag = G_U_edge_index[0] != G_U_edge_index[1]
    terms = []
    if lower_offdiag.any():
        terms.append(G_L_values[lower_offdiag].pow(2).mean())
    if upper_offdiag.any():
        terms.append(G_U_values[upper_offdiag].pow(2).mean())
    if not terms:
        return G_L_values.new_zeros(())
    return 1e-6 * torch.stack(terms).mean()


def compute_losses(args, model, node_attr, edge_attr, edge_index,
                   u_next, rhs, batch_idx, data, device,
                   validation=False):
    pred_x, pred_rhs, (G_L_ei, G_L_val), (G_U_ei, G_U_val), (A_ei, A_val) = model(
        node_attr, edge_attr, edge_index,
        diag=data.diag if hasattr(data, 'diag') else None,
        input_r=rhs,
        input_x=torch.zeros_like(u_next, device=device),
        batch_idx=batch_idx,
        include_r=args.use_r if hasattr(args, 'use_r') else False,
        use_global=args.use_global if hasattr(args, 'use_global') else False,
        diagonalize=args.diagonalize if hasattr(args, 'diagonalize') else False,
        use_pred_x=args.use_pred_x if hasattr(args, 'use_pred_x') else False)

    N = node_attr.shape[0]
    dirichlet_mask = node_attr[:, model.dirichlet_idx].to(torch.bool)
    probe_count = args.fsai_val_probes if validation else args.fsai_train_probes

    inv = fsai_inverse_probe_loss(
        G_L_ei, G_L_val, G_U_ei, G_U_val,
        A_ei, A_val, N, device,
        dirichlet_mask=dirichlet_mask,
        num_probes=probe_count,
        huber_beta=args.fsai_loss_huber_beta,
        residual_clip=args.fsai_loss_residual_clip)
    rhs_loss = fsai_rhs_loss(
        G_L_ei, G_L_val, G_U_ei, G_U_val,
        rhs=rhs, true_x=u_next, N=N,
        dirichlet_mask=dirichlet_mask,
        huber_beta=args.fsai_loss_huber_beta,
        residual_clip=args.fsai_loss_residual_clip)
    reg = fsai_value_regularization(G_L_ei, G_L_val, G_U_ei, G_U_val)

    total = inv * args.fsai_inverse_loss_weight
    total = total + rhs_loss * args.fsai_rhs_loss_weight
    total = total + reg * args.fsai_reg_weight

    x_loss = None
    if args.x_loss_weight > 1e-6:
        x_loss = F.mse_loss(pred_x, u_next, reduction='mean')
        total = total + x_loss * args.x_loss_weight

    return {
        "total": total,
        "inverse": inv,
        "rhs": rhs_loss,
        "reg": reg,
        "x": x_loss,
        "pred_x": pred_x,
        "pred_rhs": pred_rhs,
        "G_L_ei": G_L_ei,
        "G_L_val": G_L_val,
        "G_U_ei": G_U_ei,
        "G_U_val": G_U_val,
        "A_ei": A_ei,
        "A_val": A_val,
        "dirichlet_mask": dirichlet_mask,
    }


def train_epoch(args, train_loader, model, optimizer, lr_scheduler, device,
                tb_writer=None, tb_rate=1, total_steps=0, epoch=0, logger=None):
    total_loss = 0.0
    total_inv = 0.0
    total_rhs = 0.0
    total_reg = 0.0
    total_x = 0.0
    total_num = 0

    model.train()
    tqdm_itr = tqdm(train_loader, position=1, desc='Training', leave=True)

    for i, data in enumerate(tqdm_itr):
        data = data.to(device)
        optimizer.zero_grad()

        losses = compute_losses(
            args, model, data.x, data.edge_attr, data.edge_index,
            data.u_next, data.rhs, data.batch, data, device,
            validation=False)
        loss = losses["total"]
        if not torch.isfinite(loss):
            if logger is not None:
                logger.info(f'Iter {i} non-finite loss, skipping batch')
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if 'ReduceLROnPlateau' not in str(lr_scheduler.__class__):
            lr_scheduler.step()

        x_loss_val = 0.0 if losses["x"] is None else losses["x"].item()
        total_loss += loss.item()
        total_inv += losses["inverse"].item()
        total_rhs += losses["rhs"].item()
        total_reg += losses["reg"].item()
        total_x += x_loss_val
        total_num += 1
        total_steps += 1

        log_str = (f'Train loss: {loss.item():.4e} | '
                   f'inv: {losses["inverse"].item():.4e} | '
                   f'rhs: {losses["rhs"].item():.4e} | '
                   f'reg: {losses["reg"].item():.4e}')
        if x_loss_val > 0:
            log_str += f' | x: {x_loss_val:.4e}'
        tqdm_itr.set_postfix_str(log_str)

        if logger is not None and (tb_rate <= 1 or i % tb_rate == 0 or i == len(train_loader) - 1):
            logger.info(f'Iter {i} ' + log_str)

    avg_loss = total_loss / max(1, total_num)
    avg_inv = total_inv / max(1, total_num)
    avg_rhs = total_rhs / max(1, total_num)
    avg_reg = total_reg / max(1, total_num)
    avg_x = total_x / max(1, total_num)

    if tb_writer is not None:
        tb_writer.add_scalar("Loss_Train/total", avg_loss, global_step=total_steps)
        tb_writer.add_scalar("Loss_Train/inverse", avg_inv, global_step=total_steps)
        tb_writer.add_scalar("Loss_Train/rhs", avg_rhs, global_step=total_steps)
        tb_writer.add_scalar("Loss_Train/reg", avg_reg, global_step=total_steps)
        tb_writer.add_scalar("Loss_Train/x", avg_x, global_step=total_steps)

    if logger is not None:
        logger.info(
            f'Train epoch summary: total={avg_loss:.4e} | inv={avg_inv:.4e} | '
            f'rhs={avg_rhs:.4e} | reg={avg_reg:.4e} | x={avg_x:.4e} | '
            f'batches={total_num}')

    return avg_loss, total_steps


def val_epoch(args, val_loader, model, device,
              tb_writer=None, total_steps=0, epoch=0,
              save_dir='results/', logger=None):
    total_loss = 0.0
    total_inv = 0.0
    total_rhs = 0.0
    total_reg = 0.0
    total_x = 0.0
    total_num = 0

    model.eval()
    tqdm_itr = tqdm(val_loader, position=1, desc='Validation', leave=True)

    for data in tqdm_itr:
        data = data.to(device)
        with torch.no_grad():
            losses = compute_losses(
                args, model, data.x, data.edge_attr, data.edge_index,
                data.u_next, data.rhs, data.batch, data, device,
                validation=True)

        x_loss_val = 0.0 if losses["x"] is None else losses["x"].item()
        total_loss += losses["total"].item()
        total_inv += losses["inverse"].item()
        total_rhs += losses["rhs"].item()
        total_reg += losses["reg"].item()
        total_x += x_loss_val
        total_num += 1

        tqdm_itr.set_postfix_str(
            f'Val loss: {losses["total"].item():.4e} | '
            f'inv: {losses["inverse"].item():.4e} | '
            f'rhs: {losses["rhs"].item():.4e} | reg: {losses["reg"].item():.4e}')

    avg_loss = total_loss / max(1, total_num)
    avg_inv = total_inv / max(1, total_num)
    avg_rhs = total_rhs / max(1, total_num)
    avg_reg = total_reg / max(1, total_num)
    avg_x = total_x / max(1, total_num)

    if tb_writer is not None:
        tb_writer.add_scalar("Loss_Val/total", avg_loss, global_step=total_steps)
        tb_writer.add_scalar("Loss_Val/inverse", avg_inv, global_step=total_steps)
        tb_writer.add_scalar("Loss_Val/rhs", avg_rhs, global_step=total_steps)
        tb_writer.add_scalar("Loss_Val/reg", avg_reg, global_step=total_steps)
        tb_writer.add_scalar("Loss_Val/x", avg_x, global_step=total_steps)

    if logger is not None:
        logger.info(
            f'Val summary: total={avg_loss:.4e} | inv={avg_inv:.4e} | '
            f'rhs={avg_rhs:.4e} | reg={avg_reg:.4e} | x={avg_x:.4e} | '
            f'batches={total_num}')

    return 0, avg_loss
