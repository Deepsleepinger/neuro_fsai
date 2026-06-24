"""Training script for Dual-Channel Neuro-ILU preconditioner.

Usage:
    python train_neuro_ilu.py --dataset heatmultisource --mesh circle_low_res ...

The key differences from train.py:
  - Uses model_neuro_ilu.Net (dual L/U output, no symmetry enforcement)
  - Unsupervised Frobenius loss: ||LU - A|| on sparsity pattern
  - BiCGSTAB solver for validation (instead of CG)
"""

import os
import sys
import time
import importlib
from subprocess import call

import numpy as np
import torch
import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import build_parser
from dataset.suitesparse_dataset import SuiteSparseDataset
from utils.training_utils_neuro_ilu import train_epoch, val_epoch
from utils.utils import get_free_gpus, save_model, load_model, get_delta_minute, ExpLR
from utils.data_config import (heat_train_config, heat_test_config,
                               inviscidflow_train_config, inviscidflow_test_config,
                               wave_train_config, wave_test_config,
                               poisson3d_train_config, poisson3d_test_config)
from utils.visualization_utils import create_logger

# ------- config -------
parser = build_parser()

# Neuro-ILU specific arguments
parser.add_argument('--frob-loss-weight', default=1.0, type=float,
                    help='weight for the unsupervised Frobenius loss ||LU - A||')
parser.add_argument('--frob-loss-huber-beta', default=1.0, type=float,
                    help='Huber beta for robust relative Frobenius residuals')
parser.add_argument('--frob-loss-residual-clip', default=10.0, type=float,
                    help='clip relative Frobenius residuals before robust loss; <=0 disables clipping')
parser.add_argument('--operator-loss-weight', default=1.0, type=float,
                    help='weight for operator consistency loss ||LUx - Ax|| on true/random probes')
parser.add_argument('--operator-loss-huber-beta', default=1.0, type=float,
                    help='Huber beta for robust relative operator residuals')
parser.add_argument('--operator-loss-residual-clip', default=10.0, type=float,
                    help='clip relative operator residuals before robust loss; <=0 disables clipping')
parser.add_argument('--diag-barrier-weight', default=0.1, type=float,
                    help='weight for U-diagonal stability barrier')
parser.add_argument('--diag-floor-rel', default=0.1, type=float,
                    help='relative floor on |U_ii| compared to |A_ii|')
parser.add_argument('--diag-floor-abs', default=1e-3, type=float,
                    help='absolute floor on |U_ii| for diagonal stability')
parser.add_argument('--u-diag-floor-rel', default=1e-3, type=float,
                    help='relative hard floor added to modelled U diagonal magnitude')
parser.add_argument('--u-diag-floor-abs', default=1e-3, type=float,
                    help='absolute hard floor added to modelled U diagonal magnitude')
parser.add_argument('--pivot-reg-weight', default=100.0, type=float,
                    help='weight for reciprocal pivot regularization on U diagonal')
parser.add_argument('--pivot-reg-threshold', default=1e-3, type=float,
                    help='danger threshold for activating pivot barrier')
parser.add_argument('--pivot-reg-eps', default=1e-8, type=float,
                    help='epsilon used in reciprocal pivot regularization')
parser.add_argument('--inverse-loss-weight', default=0.01, type=float,
                    help='weight for implicit inverse consistency loss')
parser.add_argument('--inverse-loss-probes', default=1, type=int,
                    help='number of Rademacher probes for implicit inverse loss')
parser.add_argument('--inverse-loss-max-nodes', default=2048, type=int,
                    help='skip dense inverse loss when graph has more than this many nodes; <0 enables all')
parser.add_argument('--inverse-loss-pivot-threshold', default=1e-3, type=float,
                    help='skip inverse loss until all U pivots are above this absolute value')
parser.add_argument('--inverse-loss-huber-beta', default=1.0, type=float,
                    help='Huber beta for robust relative inverse residuals')
parser.add_argument('--inverse-loss-residual-clip', default=10.0, type=float,
                    help='clip relative inverse residuals before robust loss; <=0 disables clipping')
parser.add_argument('--operator-random-probes', default=1, type=int,
                    help='number of random probe vectors for operator consistency loss')
parser.add_argument('--train-data-dir', type=str, default=None,
                    help='path to training .npy files (for suitesparse)')
parser.add_argument('--test-data-dir', type=str, default=None,
                    help='path to test .npy files (for suitesparse)')
parser.add_argument('--val-data-num', type=int, default=-1,
                    help='validation subset size; use dataset default when < 0')
parser.add_argument('--early-stopping-patience', type=int, default=-1,
                    help='stop when best validation metric has not improved for this many epochs; <=0 disables')
parser.add_argument('--plateau-patience', type=int, default=50,
                    help='ReduceLROnPlateau patience measured in validation calls')
parser.add_argument('--plateau-factor', type=float, default=0.5,
                    help='ReduceLROnPlateau multiplicative LR decay factor')
parser.add_argument('--frob-train-max-entries', default=4096, type=int,
                    help='max number of A-pattern entries used in Frobenius loss during training; <0 uses all')
parser.add_argument('--frob-val-max-entries', default=8192, type=int,
                    help='max number of A-pattern entries used in Frobenius loss during validation; <0 uses all')

args = parser.parse_args()
args.exp_name = args.exp_name + "-neuroilu-" + time.strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join(args.save_dir, args.exp_name)
os.makedirs(save_dir, exist_ok=True)
logger = create_logger('global_logger', save_dir + '/log.txt')
logger.info(args)

num_threads = args.cpu
torch.set_num_threads(num_threads)

# ------- datasets -------
if args.dataset in ['flow', 'inviscidflowmultisource', 'flowmultisource', 'flowoutdomain', 'flowgeneralize']:
    from dataset.inviscidflow_dataset_multisource import InviscidFlowDatasetMultiSource
    inviscidflow_train_config['name'] = args.mesh
    inviscidflow_train_config['density'] = [float(x) for x in args.param.split('-')[:-1]]
    inviscidflow_test_config['name'] = args.mesh
    inviscidflow_test_config['density'] = [float(args.param.split('-')[-1])]
    print(inviscidflow_train_config, inviscidflow_test_config)
    train_dataset = InviscidFlowDatasetMultiSource(
        domain_files_path=args.train_path, config=inviscidflow_train_config,
        use_data_num=args.use_data_num,
        use_high_freq=args.high_freq, augment_edge=args.augment_edge,
        use_pred_x=args.use_pred_x, high_freq_aug=args.high_freq_aug)
    test_dataset = InviscidFlowDatasetMultiSource(
        domain_files_path=args.test_path, config=inviscidflow_test_config,
        use_data_num=args.val_data_num if args.val_data_num >= 0 else 5,
        use_high_freq=args.high_freq, augment_edge=args.augment_edge,
        use_pred_x=args.use_pred_x, high_freq_aug=args.high_freq_aug)
elif args.dataset in ['heat', 'heatmultisource', 'heatoutdomain', 'heatgeneralize']:
    from dataset.heat_dataset_multisource import HeatDatasetMultiSource
    heat_train_config['name'] = args.mesh
    heat_train_config['diffusivities'] = [float(x) for x in args.param.split('-')[:-1]]
    heat_test_config['name'] = args.mesh
    heat_test_config['diffusivities'] = [float(args.param.split('-')[-1])]
    print(heat_train_config, heat_test_config)
    train_dataset = HeatDatasetMultiSource(
        config=heat_train_config, use_data_num=args.use_data_num,
        use_high_freq=args.high_freq, augment_edge=args.augment_edge,
        use_pred_x=args.use_pred_x, high_freq_aug=args.high_freq_aug)
    test_dataset = HeatDatasetMultiSource(
        config=heat_test_config,
        use_data_num=args.val_data_num if args.val_data_num >= 0 else 5,
        use_high_freq=args.high_freq, augment_edge=args.augment_edge,
        use_pred_x=args.use_pred_x, high_freq_aug=args.high_freq_aug)
elif args.dataset in ['synthetic', 'syn']:
    from dataset.synthetic_dataset import SyntheticDataset
    poisson3d_train_config['name'] = args.mesh
    poisson3d_test_config['name'] = args.mesh
    train_dataset = SyntheticDataset(
        domain_files_path=args.train_path, config=poisson3d_train_config,
        use_data_num=args.use_data_num,
        use_high_freq=args.high_freq, augment_edge=args.augment_edge,
        use_pred_x=args.use_pred_x, high_freq_aug=args.high_freq_aug)
    test_dataset = SyntheticDataset(
        domain_files_path=args.test_path, config=poisson3d_test_config,
        use_data_num=args.val_data_num if args.val_data_num >= 0 else 5,
        use_high_freq=args.high_freq, augment_edge=args.augment_edge,
        use_pred_x=args.use_pred_x, high_freq_aug=args.high_freq_aug)
elif args.dataset in ['suitesparse', 'ss']:
    train_dataset = SuiteSparseDataset(
        args.train_data_dir, use_data_num=args.use_data_num)
    test_dataset = SuiteSparseDataset(
        args.test_data_dir,
        use_data_num=args.val_data_num if args.val_data_num >= 0 else None)
elif args.dataset in ['wave', 'wavemultisource', 'waveoutdomain', 'wavegeneralize']:
    from dataset.wave_dataset_multisource import WaveDatasetMultiSource
    wave_train_config['name'] = args.mesh
    wave_train_config['speed'] = [float(x) for x in args.param.split('-')[:-1]]
    wave_test_config['name'] = args.mesh
    wave_test_config['speed'] = [float(args.param.split('-')[-1])]
    print(wave_train_config)
    print(wave_test_config)
    train_dataset = WaveDatasetMultiSource(
        domain_files_path=args.train_path, config=wave_train_config,
        use_data_num=args.use_data_num,
        use_high_freq=args.high_freq, augment_edge=args.augment_edge,
        use_pred_x=args.use_pred_x, high_freq_aug=args.high_freq_aug)
    test_dataset = WaveDatasetMultiSource(
        domain_files_path=args.test_path, config=wave_test_config,
        use_data_num=args.val_data_num if args.val_data_num >= 0 else 5,
        use_high_freq=args.high_freq, augment_edge=args.augment_edge,
        use_pred_x=args.use_pred_x, high_freq_aug=args.high_freq_aug)
else:
    raise Exception(f'Dataset {args.dataset} not yet supported for Neuro-ILU. '
                    f'Add it to train_neuro_ilu.py.')

print(f'Train size: {len(train_dataset)}, Test size: {len(test_dataset)}')
logger.info(f'Train size: {len(train_dataset)}, Test size: {len(test_dataset)}')

node_attr_dim = train_dataset.node_attr_dim
edge_attr_dim = train_dataset.edge_attr_dim
num_edges = train_dataset.num_edges
out_node_dim = train_dataset.output_dim
out_b_dim = train_dataset.b_dim
dirichlet_idx = train_dataset.dirichlet_idx

# ------- model -------
importlib.invalidate_caches()
model_def = importlib.import_module('models.model_neuro_ilu')

model = model_def.Net(
    args,
    node_attr_dim, edge_attr_dim,
    out_node_dim, out_b_dim,
    num_edges=num_edges,
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
    dirichlet_idx=dirichlet_idx,
    norm_type=args.norm)

# ------- device -------
free_gpus = [0, 1, 2]
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    free_gpus = list(set(free_gpus).intersection(
        list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))))
device = torch.device("cuda:0") if torch.cuda.is_available() and len(free_gpus) > 0 else "cpu"
model = model.to(device)

# ------- data loaders -------
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ------- optimizer -------
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
if args.scheduler == 'ExpLR':
    lr_scheduler = ExpLR(optimizer, decay_steps=4e4, min_lr=1e-8)
elif args.scheduler == 'OneCycleLR':
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=args.epochs)
elif args.scheduler == 'ReduceLROnPlateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=args.plateau_patience, factor=args.plateau_factor, min_lr=1e-8)
elif args.scheduler == 'MultiStepLR':
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 800], gamma=0.1)
else:
    raise Exception('LR scheduler not recognized')

# ------- checkpoint -------
model, lr_scheduler, last_epoch = load_model(
    args.ckpt, 'best_val.pt', model,
    'best_val_scheduler.pt', lr_scheduler,
    'last_epoch.pt')

if args.ckpt == '':
    args.ckpt = os.path.join(save_dir, 'model')
    os.makedirs(args.ckpt, exist_ok=True)

# ------- tensorboard -------
tb_dir = os.path.join(args.save_dir, args.exp_name, 'tb')
tb_writer = None if not args.tensorboard else SummaryWriter(tb_dir)
total_steps = (last_epoch + 1) * len(train_loader)

# ------- initial validation -------
print('Running initial validation...')
best_pcg_iter, best_loss = val_epoch(
    args, test_loader, model, device,
    tb_writer=tb_writer, total_steps=0, epoch=-1,
    save_dir=save_dir, logger=logger)

best_save_metric = best_loss if best_pcg_iter == 0 else best_pcg_iter
best_epoch = last_epoch
save_model(args.ckpt,
           'best_val.pt', model,
           'best_val_scheduler.pt', lr_scheduler,
           'last_epoch.pt', best_epoch)
start_time = time.time()

# ------- training loop -------
tqdm_itr = tqdm.trange(last_epoch + 1, args.epochs, position=0, leave=True)
for i in tqdm_itr:
    tqdm_itr.set_description('Epoch')
    logger.info(f'Epoch: {i}/{args.epochs}, lr: {optimizer.param_groups[0]["lr"]}')

    train_loss, total_steps = train_epoch(
        args, train_loader, model, optimizer, lr_scheduler, device,
        tb_writer=tb_writer, tb_rate=args.log_freq,
        total_steps=total_steps, epoch=i, logger=logger)

    did_validate = False
    if i % args.val_freq == 0:
        pcg_iter, val_loss = val_epoch(
            args, test_loader, model, device,
            tb_writer=tb_writer, total_steps=total_steps, epoch=i,
            save_dir=save_dir, logger=logger)
        did_validate = True

        logger.info(f'Finish epoch {i}: Train loss={train_loss:.4e}, Val loss={val_loss:.4e}')

        save_metric = val_loss if pcg_iter == 0 else pcg_iter
        if save_metric < best_save_metric:
            best_save_metric = save_metric
            best_epoch = i
            save_model(args.ckpt,
                       'best_val.pt', model,
                       'best_val_scheduler.pt', lr_scheduler,
                       'last_epoch.pt', i)

        print('Saving checkpoint...')
        save_model(args.ckpt,
                   'latest_model.pt', model,
                   'latest_val_scheduler.pt', lr_scheduler,
                   'latest_last_epoch.pt', i)

    if 'ReduceLROnPlateau' in str(lr_scheduler.__class__) and did_validate:
        lr_scheduler.step(val_loss)
    elif 'MultiStepLR' in str(lr_scheduler.__class__):
        lr_scheduler.step()

    if args.early_stopping_patience > 0 and best_epoch >= 0:
        if i - best_epoch >= args.early_stopping_patience:
            logger.info(
                f'Early stopping at epoch {i}: best epoch {best_epoch}, '
                f'best metric {best_save_metric:.4e}')
            break

    if tb_writer is not None:
        tb_writer.add_scalar("Opt/lr", optimizer.param_groups[0]['lr'], global_step=total_steps)
        tb_writer.add_scalar("Profile/epoch_time", get_delta_minute(start_time), global_step=i)
