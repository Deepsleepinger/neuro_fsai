"""Training script for Neuro-FSAI sparse approximate inverse preconditioner."""

import importlib
import os
import time

import torch
import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import build_parser
from dataset.suitesparse_dataset import SuiteSparseDataset
from utils.training_utils_neuro_fsai import train_epoch, val_epoch
from utils.utils import save_model, load_model, get_delta_minute, ExpLR
from utils.data_config import (heat_train_config, heat_test_config,
                               inviscidflow_train_config, inviscidflow_test_config,
                               wave_train_config, wave_test_config,
                               poisson3d_train_config, poisson3d_test_config)
from utils.visualization_utils import create_logger


parser = build_parser()

parser.add_argument('--fsai-inverse-loss-weight', default=1.0, type=float,
                    help='weight for ||G_U G_L A v - v|| probe loss')
parser.add_argument('--fsai-rhs-loss-weight', default=0.1, type=float,
                    help='weight for supervised RHS probe ||G_U G_L b - x||')
parser.add_argument('--fsai-reg-weight', default=1.0, type=float,
                    help='weight for small value regularization')
parser.add_argument('--fsai-train-probes', default=4, type=int,
                    help='Rademacher probes per training graph')
parser.add_argument('--fsai-val-probes', default=8, type=int,
                    help='Rademacher probes per validation graph')
parser.add_argument('--fsai-loss-huber-beta', default=1.0, type=float,
                    help='Huber beta for relative FSAI residuals')
parser.add_argument('--fsai-loss-residual-clip', default=10.0, type=float,
                    help='clip relative FSAI residuals before robust loss; <=0 disables clipping')
parser.add_argument('--fsai-offdiag-scale', default=0.1, type=float,
                    help='initial scale cap for learned off-diagonal inverse entries')
parser.add_argument('--fsai-offdiag-basis-cap', default=1.0, type=float,
                    help='cap for inverse-diagonal off-diagonal scaling; <=0 disables')
parser.add_argument('--fsai-diag-scale', default=0.0, type=float,
                    help='deprecated; FSAI diagonal is fixed to the Jacobi residual baseline')
parser.add_argument('--fsai-diag-abs-floor', default=1e-2, type=float,
                    help='deprecated; kept for checkpoint/config compatibility')
parser.add_argument('--fsai-jacobi-eps', default=1e-12, type=float,
                    help='epsilon for residual Jacobi inverse diagonal baseline')
parser.add_argument('--fsai-relative-value-clip', default=10.0, type=float,
                    help='clip relative edge value A_ij/sqrt(|A_ii A_jj|) before encoding; <=0 disables')
parser.add_argument('--train-data-dir', type=str, default=None,
                    help='path to training .npy files for SuiteSparse')
parser.add_argument('--test-data-dir', type=str, default=None,
                    help='path to validation .npy files for SuiteSparse')
parser.add_argument('--val-data-num', type=int, default=-1,
                    help='validation subset size; use dataset default when < 0')
parser.add_argument('--early-stopping-patience', type=int, default=-1,
                    help='stop when validation metric has not improved for this many epochs; <=0 disables')
parser.add_argument('--plateau-patience', type=int, default=20,
                    help='ReduceLROnPlateau patience measured in validation calls')
parser.add_argument('--plateau-factor', type=float, default=0.5,
                    help='ReduceLROnPlateau multiplicative LR decay factor')

args = parser.parse_args()
args.exp_name = args.exp_name + "-neurofsai-" + time.strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join(args.save_dir, args.exp_name)
os.makedirs(save_dir, exist_ok=True)
logger = create_logger('global_logger', save_dir + '/log.txt')
logger.info(args)

torch.set_num_threads(args.cpu)

if args.dataset in ['flow', 'inviscidflowmultisource', 'flowmultisource', 'flowoutdomain', 'flowgeneralize']:
    from dataset.inviscidflow_dataset_multisource import InviscidFlowDatasetMultiSource
    inviscidflow_train_config['name'] = args.mesh
    inviscidflow_train_config['density'] = [float(x) for x in args.param.split('-')[:-1]]
    inviscidflow_test_config['name'] = args.mesh
    inviscidflow_test_config['density'] = [float(args.param.split('-')[-1])]
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
    train_dataset = SuiteSparseDataset(args.train_data_dir, use_data_num=args.use_data_num)
    test_dataset = SuiteSparseDataset(
        args.test_data_dir,
        use_data_num=args.val_data_num if args.val_data_num >= 0 else None)
elif args.dataset in ['wave', 'wavemultisource', 'waveoutdomain', 'wavegeneralize']:
    from dataset.wave_dataset_multisource import WaveDatasetMultiSource
    wave_train_config['name'] = args.mesh
    wave_train_config['speed'] = [float(x) for x in args.param.split('-')[:-1]]
    wave_test_config['name'] = args.mesh
    wave_test_config['speed'] = [float(args.param.split('-')[-1])]
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
    raise Exception(f'Dataset {args.dataset} not supported for Neuro-FSAI.')

print(f'Train size: {len(train_dataset)}, Test size: {len(test_dataset)}')
logger.info(f'Train size: {len(train_dataset)}, Test size: {len(test_dataset)}')

node_attr_dim = train_dataset.node_attr_dim
edge_attr_dim = train_dataset.edge_attr_dim
num_edges = train_dataset.num_edges
out_node_dim = train_dataset.output_dim
out_b_dim = train_dataset.b_dim
dirichlet_idx = train_dataset.dirichlet_idx

importlib.invalidate_caches()
model_def = importlib.import_module('models.model_neuro_fsai')
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

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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

model, lr_scheduler, last_epoch = load_model(
    args.ckpt, 'best_val.pt', model,
    'best_val_scheduler.pt', lr_scheduler,
    'last_epoch.pt')

if args.ckpt == '':
    args.ckpt = os.path.join(save_dir, 'model')
    os.makedirs(args.ckpt, exist_ok=True)

tb_dir = os.path.join(args.save_dir, args.exp_name, 'tb')
tb_writer = None if not args.tensorboard else SummaryWriter(tb_dir)
total_steps = (last_epoch + 1) * len(train_loader)

print('Running initial validation...')
best_pcg_iter, best_loss = val_epoch(
    args, test_loader, model, device,
    tb_writer=tb_writer, total_steps=0, epoch=-1,
    save_dir=save_dir, logger=logger)

best_save_metric = best_loss
best_epoch = max(last_epoch, -1)
save_model(args.ckpt,
           'best_val.pt', model,
           'best_val_scheduler.pt', lr_scheduler,
           'last_epoch.pt', best_epoch)
start_time = time.time()

tqdm_itr = tqdm.trange(last_epoch + 1, args.epochs, position=0, leave=True)
for i in tqdm_itr:
    tqdm_itr.set_description('Epoch')
    logger.info(f'Epoch: {i}/{args.epochs}, lr: {optimizer.param_groups[0]["lr"]}')

    train_loss, total_steps = train_epoch(
        args, train_loader, model, optimizer, lr_scheduler, device,
        tb_writer=tb_writer, tb_rate=args.log_freq,
        total_steps=total_steps, epoch=i, logger=logger)

    did_validate = False
    val_loss = None
    if i % args.val_freq == 0:
        _, val_loss = val_epoch(
            args, test_loader, model, device,
            tb_writer=tb_writer, total_steps=total_steps, epoch=i,
            save_dir=save_dir, logger=logger)
        did_validate = True

        logger.info(f'Finish epoch {i}: Train loss={train_loss:.4e}, Val loss={val_loss:.4e}')

        if val_loss < best_save_metric:
            best_save_metric = val_loss
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
        old_lr = optimizer.param_groups[0]['lr']
        lr_scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            logger.info(
                f'Learning rate reduced: {old_lr:.4e} -> {new_lr:.4e}. '
                'Reloading best validation model weights only.')
            best_ckpt_path = os.path.join(args.ckpt, 'best_val.pt')
            if os.path.exists(best_ckpt_path):
                checkpoint = torch.load(best_ckpt_path, map_location=device)
                state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
                model.load_state_dict(state_dict)
                logger.info(f'Reloaded best validation weights from {best_ckpt_path}')
            else:
                logger.warning(f'Best validation checkpoint not found: {best_ckpt_path}')
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
