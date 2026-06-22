# Neuro-SPAI Current Model Package

Created from workspace:

`/home/predator/0pde_ai/mysc/Preconditioner_neuroilu_fix_v1`

Python environment:

`/home/predator/anaconda3/envs/gcn-spcg/bin/python`

## Model Line

Current active research line is Neuro-SPAI:

`y = G @ r`

The model predicts one sparse approximate inverse matrix `G`.  Training labels
are supervised SPILU inverse `row-top256` edge values.  Validation is based on
BiCGSTAB iterations, not edge-value loss alone.

## Included Code

- `scripts/train_neuro_spai_single.py`
- `scripts/train_neuro_spai_multi.py`
- `scripts/evaluate_spilu_inverse_topk_single.py`
- `scripts/evaluate_spilu_action_oracle_single.py`
- `models/model_neuro_fsai.py` for `PDEDirectedConv`
- `utils/convert_suitesparse.py`
- `utils/topology_expansion.py`
- `dataset/suitesparse_dataset.py`
- `pcg.py`

## Included Checkpoints

Single-matrix overfit with node embedding:

| Matrix | Jacobi | Neuro-SPAI Best | Directory |
|---|---:|---:|---|
| `DRIVCAV_cavity05` | 394 | 101 | `single_neuro_spai_embed_h64_lr1e3_e1000-rowtop256-20260622-103356` |
| `Bai_cdde1` | 86 | 66 | `single_neuro_spai_cdde1_embed_h64_e1000-rowtop256-20260622-103755` |
| `Bomhof_circuit_1` | 183 | 45 | `single_neuro_spai_circuit1_embed_h64_e1000-rowtop256-20260622-103843` |

Shared no-node-embedding experiments:

| Run | Result |
|---|---|
| `multi_neuro_spai_3mat_noembed_h128_e500-rowtop256-20260622-104246` | Improves `cdde1` and `circuit_1`, fails `cavity05` |
| `multi_neuro_spai_3mat_noembed_h64_mp1_e300-rowtop256-20260622-104550` | Strong `circuit_1` result, still fails `cavity05` |

## Main Conclusion

Single-matrix Neuro-SPAI is viable and beats Jacobi on 3/3 tested matrices.
Shared cross-category training is not stable yet.  Next work should train
separate category models, especially `circuit` and `flow`, instead of mixing
all physics classes in one model.

## Notes

This package intentionally excludes raw SuiteSparse archives, prepared datasets,
large result folders, and `__pycache__`.
