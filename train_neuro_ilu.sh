#!/bin/bash
# Dual-Channel Neuro-ILU training script
#
# Key differences from train.sh:
#   - Uses model_neuro_ilu (no symmetry enforcement, dual L/U output)
#   - Unsupervised Frobenius loss: ||LU - A|| on sparsity pattern (no ILU labels needed)
#   - BiCGSTAB solver for validation

python train_neuro_ilu.py \
    --use-data-num 2000 \
    --batch-size 8 \
    --mesh 'circle_low_res' \
    --param '100.0-100.0' \
    --num-iterations 5 \
    --hidden-layers-encoder 1 \
    --hidden-layers-decoder 1 \
    --hidden-layers-processor 1 \
    --hidden-dim 16 \
    --lr 1e-3 \
    --epochs 5000 \
    --dataset heatmultisource \
    --tensorboard \
    --frob-loss-weight 1.0 \
    --x-loss-weight 0.1 \
    --simulate \
    --val-freq 10 \
    --use-pred-x \
    --exp-name 'neuro_ilu_exp1'
