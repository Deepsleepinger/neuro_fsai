import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import bicgstab, spilu


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_neuro_spai_single import (
    NeuroSPAI,
    apply_reordering,
    build_graph_tensors,
    build_spilu_inverse_row_topk,
    decode_residual,
    evaluate_values,
    hutchinson_loss,
    load_rhs,
    max_abs_equilibrate,
    metric_key,
    parse_float_list,
    run_bicgstab,
)
from utils.convert_suitesparse import canonicalize_sparse_matrix, read_mtx_from_tar


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one shared Neuro-SPAI model on multiple SPILU inverse top-k targets.")
    parser.add_argument("--matrix-tars", nargs="+", required=True)
    parser.add_argument(
        "--eval-matrix-tars",
        nargs="*",
        default=None,
        help="Optional eval-only matrices. If omitted, train matrices are also evaluated.")
    parser.add_argument("--save-dir", default="results/local_checkpoints")
    parser.add_argument("--exp-name", default="multi_neuro_spai")
    parser.add_argument("--row-topk", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-iterations", type=int, default=0)
    parser.add_argument("--decoder-type", choices=["mlp", "bilinear"], default="mlp")
    parser.add_argument("--weight-abs", type=float, default=5.0)
    parser.add_argument("--eval-damping-grid", default="1.0,0.5,0.25,0.1,0.05,0.01,0.005,0.001")
    parser.add_argument("--feature-mode", choices=["algebraic", "legacy"], default="algebraic")
    parser.add_argument("--reorder", choices=["none", "rcm"], default="none")
    parser.add_argument("--equilibrate", action="store_true")
    parser.add_argument("--equil-iters", type=int, default=5)
    parser.add_argument("--equil-eps", type=float, default=1e-12)
    parser.add_argument("--spectral-pe-dim", type=int, default=0)
    parser.add_argument("--topology-hop", type=int, choices=[1, 2], default=1)
    parser.add_argument("--topology-drop-tol", type=float, default=0.0)
    parser.add_argument("--topology-row-topk", type=int, default=64)
    parser.add_argument("--amg-levels", type=int, default=0)
    parser.add_argument("--amg-min-coarse-nodes", type=int, default=500)
    parser.add_argument("--target-transform", choices=["linear", "signed_log10"], default="linear")
    parser.add_argument("--target-scale-mode", choices=["teacher", "jacobi", "unit"], default="teacher")
    parser.add_argument("--base-mode", choices=["jacobi", "identity", "zero"], default="jacobi")
    parser.add_argument("--log-output-clip", type=float, default=16.0)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--hutchinson-weight", type=float, default=0.0)
    parser.add_argument("--teacher-hutchinson-weight", type=float, default=0.0)
    parser.add_argument("--ssl-hutchinson-weight", type=float, default=1.0)
    parser.add_argument("--hutchinson-probes", type=int, default=4)
    parser.add_argument(
        "--hutchinson-probe-mode",
        choices=["random", "krylov_residual", "mixed"],
        default="random",
        help="Use random Rademacher probes, stalled Krylov residual probes, or both.")
    parser.add_argument(
        "--krylov-probe-steps",
        type=int,
        default=5,
        help="Unpreconditioned BiCGSTAB steps used to build stalled residual probes.")
    parser.add_argument(
        "--teacher-max-n",
        type=int,
        default=0,
        help="If >0, train matrices with N > teacher_max_n skip dense teacher and use SSL loss only.")
    parser.add_argument(
        "--poison-train-replicas",
        type=int,
        default=0,
        help="Add this many fixed poisoned SSL replicas for each eligible training matrix.")
    parser.add_argument("--poison-ratio", type=float, default=0.10)
    parser.add_argument("--poison-diag-penalty", type=float, default=1e-4)
    parser.add_argument(
        "--poison-offdiag-noise",
        type=float,
        default=0.0,
        help="Relative Gaussian perturbation applied to off-diagonal entries in poisoned rows.")
    parser.add_argument(
        "--poison-max-n",
        type=int,
        default=2000,
        help="Only poison train matrices with N <= this value; set 0 to disable the size cap.")
    parser.add_argument("--poison-seed", type=int, default=20260623)
    parser.add_argument("--val-freq", type=int, default=100)
    parser.add_argument("--log-freq", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rhs-seed", type=int, default=20260622)
    parser.add_argument("--spilu-drop-tol", type=float, default=1e-4)
    parser.add_argument("--spilu-fill-factor", type=float, default=10.0)
    parser.add_argument(
        "--eval-no-teacher",
        action="store_true",
        help="Build eval graphs without dense SPILU inverse labels; required for large zero-shot targets.")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--rtol", type=float, default=1e-8)
    return parser.parse_args()


def matrix_name(path):
    return pathlib.Path(path).name.replace(".tar.gz", "")


def stable_path_seed(path):
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def poison_matrix(A_csr, poison_ratio, diag_penalty, offdiag_noise=0.0, seed=0):
    """Inject local diagonal weakness and optional asymmetric row perturbations."""
    A = A_csr.tocsr(copy=True).astype(np.float64)
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    num_poison = int(round(n * max(0.0, min(1.0, poison_ratio))))
    if num_poison <= 0 or n == 0:
        return A, {
            "enabled": False,
            "num_poison_rows": 0,
            "diag_found": 0,
            "seed": int(seed),
        }

    target_rows = rng.choice(n, size=min(num_poison, n), replace=False)
    diag_found = 0
    offdiag_touched = 0
    for row in target_rows:
        start, end = A.indptr[row], A.indptr[row + 1]
        cols = A.indices[start:end]
        diag_loc = np.flatnonzero(cols == row)
        if diag_loc.size:
            A.data[start + int(diag_loc[0])] *= diag_penalty
            diag_found += 1
        if offdiag_noise > 0.0 and end > start:
            local = np.arange(start, end)
            offdiag_mask = cols != row
            offdiag_idx = local[offdiag_mask]
            if offdiag_idx.size:
                noise = rng.normal(
                    loc=0.0,
                    scale=offdiag_noise,
                    size=offdiag_idx.shape[0],
                )
                A.data[offdiag_idx] *= np.clip(1.0 + noise, 0.1, 10.0)
                offdiag_touched += int(offdiag_idx.shape[0])

    stats = {
        "enabled": True,
        "num_poison_rows": int(target_rows.shape[0]),
        "poison_ratio": float(poison_ratio),
        "diag_penalty": float(diag_penalty),
        "offdiag_noise": float(offdiag_noise),
        "diag_found": int(diag_found),
        "offdiag_touched": int(offdiag_touched),
        "seed": int(seed),
    }
    A.sum_duplicates()
    A.eliminate_zeros()
    return A, stats


def build_krylov_residual_probe(A_csr, rhs, max_steps):
    """Return a stalled unpreconditioned BiCGSTAB residual for algebraic SSL."""
    start = time.perf_counter()
    rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    norm_rhs = max(1e-30, float(np.linalg.norm(rhs)))
    counter = {"iters": 0}

    def callback(_xk):
        counter["iters"] += 1

    try:
        x, info = bicgstab(
            A_csr,
            rhs,
            x0=np.zeros(A_csr.shape[0], dtype=np.float64),
            rtol=0.0,
            atol=0.0,
            maxiter=max(1, int(max_steps)),
            callback=callback,
        )
        if not np.all(np.isfinite(x)):
            raise FloatingPointError("non-finite Krylov iterate")
        residual = rhs - A_csr @ x
        norm_residual = float(np.linalg.norm(residual))
        if not np.isfinite(norm_residual) or norm_residual <= 1e-30:
            raise FloatingPointError("invalid or near-zero Krylov residual")
        fallback = False
        error = None
    except Exception as exc:  # Keep graph construction robust for pathological matrices.
        residual = rhs.copy()
        norm_residual = float(np.linalg.norm(residual))
        info = None
        fallback = True
        error = str(exc)

    stats = {
        "mode": "krylov_residual",
        "steps_requested": int(max_steps),
        "raw_callback_iterations": int(counter["iters"]),
        "info": None if info is None else int(info),
        "rhs_norm": float(norm_rhs),
        "residual_norm": float(norm_residual),
        "relative_residual": float(norm_residual / norm_rhs),
        "fallback": bool(fallback),
        "error": error,
        "build_time": float(time.perf_counter() - start),
    }
    return residual.astype(np.float32, copy=False), stats


def algebraic_ssl_loss(pred_norm, graph, args, generator):
    if args.hutchinson_probe_mode == "random":
        return hutchinson_loss(
            pred_norm, graph, args.log_output_clip,
            args.hutchinson_probes, generator)
    if args.hutchinson_probe_mode == "krylov_residual":
        return hutchinson_loss(
            pred_norm, graph, args.log_output_clip,
            args.hutchinson_probes, generator,
            probe_vectors=graph.get("krylov_probe_vectors"))
    if args.hutchinson_probe_mode == "mixed":
        random_loss = hutchinson_loss(
            pred_norm, graph, args.log_output_clip,
            args.hutchinson_probes, generator)
        residual_loss = hutchinson_loss(
            pred_norm, graph, args.log_output_clip,
            args.hutchinson_probes, generator,
            probe_vectors=graph.get("krylov_probe_vectors"))
        return 0.5 * (random_loss + residual_loss)
    raise ValueError(f"unknown hutchinson_probe_mode={args.hutchinson_probe_mode!r}")


def build_case(tar_path, args, device, need_teacher=True, poison_config=None,
               name_suffix=""):
    matrix = canonicalize_sparse_matrix(read_mtx_from_tar(tar_path))
    poison_stats = None
    if poison_config is not None:
        matrix, poison_stats = poison_matrix(
            matrix,
            poison_ratio=poison_config["ratio"],
            diag_penalty=poison_config["diag_penalty"],
            offdiag_noise=poison_config["offdiag_noise"],
            seed=poison_config["seed"],
        )
    has_teacher = bool(need_teacher)
    if has_teacher and args.teacher_max_n > 0 and matrix.shape[0] > args.teacher_max_n:
        has_teacher = False
    scale = max(1.0, float(np.max(np.abs(matrix.data))) if matrix.nnz else 1.0)
    A_scaled = (matrix / scale).astype(np.float64).tocsr()
    b = load_rhs(None, scale, A_scaled, args.rhs_seed)
    A_scaled, b, _, reorder_stats = apply_reordering(A_scaled, b, args.reorder)
    if args.equilibrate:
        A_model, dr, dc, equil_stats = max_abs_equilibrate(
            A_scaled, args.equil_iters, args.equil_eps)
    else:
        A_model = A_scaled
        dr = np.ones(A_scaled.shape[0], dtype=np.float64)
        dc = np.ones(A_scaled.shape[0], dtype=np.float64)
        equil_stats = None

    teacher_error = None
    if has_teacher:
        try:
            target_csr, teacher, dense_inverse = build_spilu_inverse_row_topk(
                A_model, args.row_topk, args.spilu_drop_tol, args.spilu_fill_factor)
            inverse_absmax = float(np.abs(dense_inverse).max())
        except RuntimeError as exc:
            has_teacher = False
            target_csr = sp.csr_matrix(A_model.shape, dtype=np.float64)
            teacher = None
            inverse_absmax = None
            teacher_error = str(exc)
    else:
        target_csr = sp.csr_matrix(A_model.shape, dtype=np.float64)
        try:
            teacher = spilu(
                A_model.tocsc(),
                drop_tol=args.spilu_drop_tol,
                fill_factor=args.spilu_fill_factor,
            )
        except RuntimeError as exc:
            teacher = None
            teacher_error = str(exc)
        inverse_absmax = None
    graph = build_graph_tensors(
        A_model, target_csr, device,
        feature_mode=args.feature_mode,
        target_transform=args.target_transform,
        topology_hop=args.topology_hop,
        topology_drop_tol=args.topology_drop_tol,
        topology_row_topk=args.topology_row_topk,
        spectral_pe_dim=args.spectral_pe_dim,
        target_scale_mode=args.target_scale_mode,
        base_mode=args.base_mode,
        amg_levels=args.amg_levels,
        amg_min_coarse_nodes=args.amg_min_coarse_nodes)
    krylov_probe_stats = None
    if args.hutchinson_probe_mode in {"krylov_residual", "mixed"}:
        rhs_model = dr * b if args.equilibrate else b
        probe, krylov_probe_stats = build_krylov_residual_probe(
            A_model, rhs_model, args.krylov_probe_steps)
        graph["krylov_probe_vectors"] = torch.from_numpy(probe[:, None]).to(device)
    recovery_values = (dc[graph["row"]] * dr[graph["col"]]).astype(np.float64, copy=False)
    diag = A_scaled.diagonal().astype(np.float64)
    diag[diag == 0.0] = 1.0
    if teacher is not None and args.equilibrate:
        spilu_apply = lambda v: dc * teacher.solve(dr * v)
    elif teacher is not None:
        spilu_apply = teacher.solve
    else:
        spilu_apply = None
    baselines = {
        "identity": run_bicgstab(A_scaled, b, lambda v: v, args.max_iter, args.rtol),
        "jacobi": run_bicgstab(A_scaled, b, lambda v: v / diag, args.max_iter, args.rtol),
        "spilu": (
            None if spilu_apply is None
            else run_bicgstab(A_scaled, b, spilu_apply, args.max_iter, args.rtol)
        ),
    }
    if teacher_error is not None:
        baselines["spilu_error"] = teacher_error
    if has_teacher:
        target_result, _ = evaluate_values(
            A_scaled, b, graph["target_values"] * recovery_values,
            graph, args.max_iter, args.rtol)
        baselines["target_spai"] = target_result
    else:
        baselines["target_spai"] = None
    return {
        "name": matrix_name(tar_path) + name_suffix,
        "tar_path": tar_path,
        "A_scaled": A_scaled,
        "A_model": A_model,
        "b": b,
        "graph": graph,
        "has_teacher": has_teacher,
        "recovery_values": recovery_values,
        "weights": 1.0 + args.weight_abs * graph["target_norm"].abs(),
        "baselines": baselines,
        "target_nnz": int(target_csr.nnz),
        "support_nnz": int(graph["support_nnz"]),
        "inverse_absmax": inverse_absmax,
        "scale": float(scale),
        "reordering": reorder_stats,
        "equilibration": equil_stats,
        "krylov_probe_stats": krylov_probe_stats,
        "poison_stats": poison_stats,
        "teacher_error": teacher_error,
    }


def evaluate_case(model, case, damping_grid, args):
    graph = case["graph"]
    model.eval()
    with torch.no_grad():
        pred_norm = model(
            graph["node_attr"], graph["edge_index"], graph["edge_attr"],
            graph["amg_data"])
        pred_residual_t = decode_residual(pred_norm, graph, args.log_output_clip)
        pred_residual = pred_residual_t.detach().cpu().numpy().astype(np.float64)
    best_result = None
    best_values = None
    best_alpha = None
    for alpha in damping_grid:
        values_hat = graph["base_values64"] + alpha * pred_residual
        values = values_hat * case["recovery_values"]
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
    train_cases = []
    for tar_path in args.matrix_tars:
        start = time.perf_counter()
        case = build_case(tar_path, args, device, need_teacher=True)
        train_cases.append(case)
        log(
            f"train_case={case['name']} N={case['A_scaled'].shape[0]} "
            f"A_nnz={case['A_scaled'].nnz} support_nnz={case['support_nnz']} "
            f"target_nnz={case['target_nnz']} has_teacher={case['has_teacher']} "
            f"topology_hop={args.topology_hop} "
            f"topology_drop_tol={args.topology_drop_tol:.3e} "
            f"topology_row_topk={args.topology_row_topk} "
            f"amg_levels={len(case['graph']['amg_data'])} "
            f"reorder={args.reorder} spectral_pe_dim={args.spectral_pe_dim} "
            f"equilibrate={args.equilibrate} "
            f"target_scale_mode={args.target_scale_mode} "
            f"base_mode={args.base_mode} "
            f"hutchinson_probe_mode={args.hutchinson_probe_mode} "
            f"krylov_probe_stats={json.dumps(case['krylov_probe_stats'])} "
            f"poison_stats={json.dumps(case['poison_stats'])} "
            f"target_scale={case['graph']['target_scale']:.6e} "
            f"build_time={time.perf_counter() - start:.3f}s "
            f"baselines={json.dumps(case['baselines'])}")
        poison_allowed = (
            args.poison_train_replicas > 0
            and (args.poison_max_n <= 0 or case["A_scaled"].shape[0] <= args.poison_max_n)
        )
        if poison_allowed:
            base_seed = args.poison_seed + stable_path_seed(tar_path)
            for rep in range(args.poison_train_replicas):
                start = time.perf_counter()
                poison_config = {
                    "ratio": args.poison_ratio,
                    "diag_penalty": args.poison_diag_penalty,
                    "offdiag_noise": args.poison_offdiag_noise,
                    "seed": base_seed + rep * 1000003,
                }
                poisoned = build_case(
                    tar_path,
                    args,
                    device,
                    need_teacher=False,
                    poison_config=poison_config,
                    name_suffix=f"_poison{rep:02d}",
                )
                train_cases.append(poisoned)
                log(
                    f"train_case={poisoned['name']} N={poisoned['A_scaled'].shape[0]} "
                    f"A_nnz={poisoned['A_scaled'].nnz} support_nnz={poisoned['support_nnz']} "
                    f"target_nnz={poisoned['target_nnz']} has_teacher={poisoned['has_teacher']} "
                    f"topology_hop={args.topology_hop} "
                    f"topology_drop_tol={args.topology_drop_tol:.3e} "
                    f"topology_row_topk={args.topology_row_topk} "
                    f"amg_levels={len(poisoned['graph']['amg_data'])} "
                    f"reorder={args.reorder} spectral_pe_dim={args.spectral_pe_dim} "
                    f"equilibrate={args.equilibrate} "
                    f"target_scale_mode={args.target_scale_mode} "
                    f"base_mode={args.base_mode} "
                    f"hutchinson_probe_mode={args.hutchinson_probe_mode} "
                    f"krylov_probe_stats={json.dumps(poisoned['krylov_probe_stats'])} "
                    f"poison_stats={json.dumps(poisoned['poison_stats'])} "
                    f"target_scale={poisoned['graph']['target_scale']:.6e} "
                    f"build_time={time.perf_counter() - start:.3f}s "
                    f"baselines={json.dumps(poisoned['baselines'])}")
    eval_cases = []
    for tar_path in (args.eval_matrix_tars or []):
        start = time.perf_counter()
        case = build_case(tar_path, args, device, need_teacher=not args.eval_no_teacher)
        eval_cases.append(case)
        log(
            f"eval_case={case['name']} N={case['A_scaled'].shape[0]} "
            f"A_nnz={case['A_scaled'].nnz} support_nnz={case['support_nnz']} "
            f"target_nnz={case['target_nnz']} has_teacher={case['has_teacher']} "
            f"topology_hop={args.topology_hop} "
            f"topology_drop_tol={args.topology_drop_tol:.3e} "
            f"topology_row_topk={args.topology_row_topk} "
            f"amg_levels={len(case['graph']['amg_data'])} "
            f"reorder={args.reorder} spectral_pe_dim={args.spectral_pe_dim} "
            f"equilibrate={args.equilibrate} eval_no_teacher={args.eval_no_teacher} "
            f"target_scale_mode={args.target_scale_mode} "
            f"base_mode={args.base_mode} "
            f"hutchinson_probe_mode={args.hutchinson_probe_mode} "
            f"krylov_probe_stats={json.dumps(case['krylov_probe_stats'])} "
            f"poison_stats={json.dumps(case['poison_stats'])} "
            f"target_scale={case['graph']['target_scale']:.6e} "
            f"build_time={time.perf_counter() - start:.3f}s "
            f"baselines={json.dumps(case['baselines'])}")
    metric_cases = eval_cases or train_cases

    first = train_cases[0]["graph"]
    model = NeuroSPAI(
        node_dim=first["node_attr"].shape[1],
        edge_dim=first["edge_attr"].shape[1],
        hidden_dim=args.hidden_dim,
        num_iterations=args.num_iterations,
        use_node_embedding=False,
        decoder_type=args.decoder_type,
        amg_levels=args.amg_levels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    damping_grid = parse_float_list(args.eval_damping_grid)
    hutchinson_gen = torch.Generator(device=device)
    hutchinson_gen.manual_seed(args.seed + 17)

    best_metric = float("inf")
    best_epoch = -1
    best_results = None
    latest_results = None
    for epoch in range(args.epochs + 1):
        loss_values = []
        graph_maes = []
        if epoch > 0:
            optimizer.zero_grad(set_to_none=True)
        for case in train_cases:
            model.train()
            graph = case["graph"]
            pred_norm = model(
                graph["node_attr"], graph["edge_index"], graph["edge_attr"],
                graph["amg_data"])
            if case["has_teacher"]:
                sq = (pred_norm - graph["target_norm"]).pow(2)
                mse_loss = (sq * case["weights"]).mean()
                mae = (pred_norm - graph["target_norm"]).abs().mean()
                teacher_hutch_weight = args.hutchinson_weight + args.teacher_hutchinson_weight
                alg_loss = pred_norm.new_zeros(())
                if teacher_hutch_weight > 0:
                    alg_loss = algebraic_ssl_loss(
                        pred_norm, graph, args, hutchinson_gen)
                loss = args.mse_weight * mse_loss + teacher_hutch_weight * alg_loss
            else:
                alg_loss = algebraic_ssl_loss(
                    pred_norm, graph, args, hutchinson_gen)
                loss = args.ssl_hutchinson_weight * alg_loss
                mae = pred_norm.abs().mean()
            loss_values.append(loss.detach())
            graph_maes.append(mae.detach())
            if epoch > 0:
                (loss / max(1, len(train_cases))).backward()

        final_loss = torch.stack(loss_values).mean()
        final_mae = torch.stack(graph_maes).mean()
        if epoch > 0:
            optimizer.step()

        should_eval = epoch == 0 or epoch == args.epochs or epoch % args.val_freq == 0
        if should_eval:
            results = {}
            for case in metric_cases:
                result, alpha, values = evaluate_case(model, case, damping_grid, args)
                results[case["name"]] = {"result": result, "alpha": alpha}
                if best_results is None:
                    continue
            metric = aggregate_metric(results, metric_cases)
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
                f"epoch={epoch:04d} loss={final_loss.item():.6e} "
                f"mae={final_mae.item():.6e} metric={metric:.6e} "
                f"best_epoch={best_epoch} results={json.dumps(compact)}")
        elif epoch % args.log_freq == 0:
            log(
                f"epoch={epoch:04d} loss={final_loss.item():.6e} "
                f"mae={final_mae.item():.6e} best_epoch={best_epoch}")

    torch.save(model.state_dict(), model_dir / "latest_model.pt")
    meta = {
        "args": vars(args),
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_results": best_results,
        "latest_results": latest_results,
        "train_cases": [
            {
                "name": case["name"],
                "tar_path": case["tar_path"],
                "N": int(case["A_scaled"].shape[0]),
                "A_nnz": int(case["A_scaled"].nnz),
                "target_nnz": case["target_nnz"],
                "has_teacher": case["has_teacher"],
                "support_nnz": case["support_nnz"],
                "topology_drop_tol": float(args.topology_drop_tol),
                "topology_row_topk": int(args.topology_row_topk),
                "reordering": case["reordering"],
                "equilibration": case["equilibration"],
                "spectral_pe_dim": int(args.spectral_pe_dim),
                "amg_levels": len(case["graph"]["amg_data"]),
                "amg_level_shapes": case["graph"]["amg_level_shapes"],
                "target_scale_mode": args.target_scale_mode,
                "target_scale": float(case["graph"]["target_scale"]),
                "base_mode": args.base_mode,
                "hutchinson_probe_mode": args.hutchinson_probe_mode,
                "krylov_probe_stats": case["krylov_probe_stats"],
                "poison_stats": case["poison_stats"],
                "teacher_error": case["teacher_error"],
                "baselines": case["baselines"],
            }
            for case in train_cases
        ],
        "eval_cases": [
            {
                "name": case["name"],
                "tar_path": case["tar_path"],
                "N": int(case["A_scaled"].shape[0]),
                "A_nnz": int(case["A_scaled"].nnz),
                "target_nnz": case["target_nnz"],
                "has_teacher": case["has_teacher"],
                "support_nnz": case["support_nnz"],
                "topology_drop_tol": float(args.topology_drop_tol),
                "topology_row_topk": int(args.topology_row_topk),
                "reordering": case["reordering"],
                "equilibration": case["equilibration"],
                "spectral_pe_dim": int(args.spectral_pe_dim),
                "eval_no_teacher": bool(args.eval_no_teacher),
                "amg_levels": len(case["graph"]["amg_data"]),
                "amg_level_shapes": case["graph"]["amg_level_shapes"],
                "target_scale_mode": args.target_scale_mode,
                "target_scale": float(case["graph"]["target_scale"]),
                "base_mode": args.base_mode,
                "hutchinson_probe_mode": args.hutchinson_probe_mode,
                "krylov_probe_stats": case["krylov_probe_stats"],
                "poison_stats": case["poison_stats"],
                "teacher_error": case["teacher_error"],
                "baselines": case["baselines"],
            }
            for case in eval_cases
        ],
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log("summary=" + json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
