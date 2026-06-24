import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spilu

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.convert_suitesparse import (
    canonicalize_sparse_matrix,
    matrix_to_graph,
    read_mtx_from_tar,
    values_on_pattern,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-tar", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--topk-fill", type=int, default=100000)
    parser.add_argument("--drop-tol", type=float, default=1e-4)
    parser.add_argument("--fill-factor", type=float, default=10.0)
    return parser.parse_args()


def graph_from_topology(A, topology, name, meta_extra):
    A = canonicalize_sparse_matrix(A)
    topology = canonicalize_sparse_matrix(topology)
    N = A.shape[0]
    topology_coo = topology.tocoo()
    row = topology_coo.row.astype(np.int64, copy=False)
    col = topology_coo.col.astype(np.int64, copy=False)
    values = values_on_pattern(A, row, col)

    graph = matrix_to_graph(A, name=name, topology_hop=1)
    edge_index = np.stack([col, row], axis=0).astype(np.int64)
    edge_attr = np.zeros((edge_index.shape[1], 3), dtype=np.float32)
    edge_attr[:, 0] = (np.abs(row - col).astype(np.float32) / N) * 2 - 1
    edge_attr[:, 1] = values.astype(np.float32)
    edge_attr[:, 2] = (row == col).astype(np.float32)

    graph["edge_index"] = edge_index
    graph["edge_attr"] = edge_attr
    graph["r"] = np.where(row != col, np.abs(values), 0.0).astype(np.float32).reshape(-1, 1)
    r_max = max(1e-8, float(np.max(np.abs(graph["r"]))) if graph["r"].size else 0.0)
    graph["r"] = graph["r"] / r_max
    r_norm = max(1e-8, float(np.linalg.norm(graph["r"].astype(np.float64))))
    graph["r"] = graph["r"] / r_norm
    graph["meta"].update({
        "E": int(edge_index.shape[1]),
        "topology_hop": 0,
        "topology_E": int(edge_index.shape[1]),
        "topology_expanded": True,
        **meta_extra,
    })
    return graph


def main():
    args = parse_args()
    matrix_path = Path(args.matrix_tar)
    name = args.name or matrix_path.name.replace(".tar.gz", "")
    output_root = Path(args.output_root)
    train_dir = output_root / "train"
    val_dir = output_root / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    A = canonicalize_sparse_matrix(read_mtx_from_tar(str(matrix_path)))
    N = A.shape[0]
    scale = max(1.0, float(np.max(np.abs(A.data))) if A.nnz else 1.0)
    A_scaled = (A / scale).tocsc()
    teacher = spilu(A_scaled, drop_tol=args.drop_tol, fill_factor=args.fill_factor)

    A_coo = A.tocoo()
    original_keys = set((A_coo.row.astype(np.int64) * N + A_coo.col.astype(np.int64)).tolist())
    fill_entries = []
    for factor in [teacher.L.tocoo(), teacher.U.tocoo()]:
        for r, c, v in zip(factor.row, factor.col, factor.data):
            if r == c:
                continue
            key = int(r) * N + int(c)
            if key in original_keys:
                continue
            fill_entries.append((abs(float(v)), int(r), int(c)))

    fill_entries.sort(reverse=True, key=lambda x: x[0])
    selected = fill_entries[:args.topk_fill]
    top_rows = np.array([r for _, r, _ in selected], dtype=np.int64)
    top_cols = np.array([c for _, _, c in selected], dtype=np.int64)

    if selected:
        fill_topology = sp.coo_matrix(
            (np.ones(len(selected), dtype=np.float32), (top_rows, top_cols)),
            shape=A.shape).tocsr()
        topology = ((A != 0) + (fill_topology != 0)).astype(np.float32).tocsr()
    else:
        topology = (A != 0).astype(np.float32).tocsr()

    graph = graph_from_topology(
        A,
        topology,
        name=name,
        meta_extra={
            "teacher": "spilu",
            "spilu_drop_tol": args.drop_tol,
            "spilu_fill_factor": args.fill_factor,
            "teacher_fill_candidates": int(len(fill_entries)),
            "teacher_fill_topk": int(args.topk_fill),
            "teacher_fill_selected": int(len(selected)),
            "teacher_L_nnz": int(teacher.L.nnz),
            "teacher_U_nnz": int(teacher.U.nnz),
            "original_nnz": int(A.nnz),
        },
    )

    np.save(train_dir / "train_0000.npy", np.array([graph], dtype=object))
    np.save(val_dir / "val_0000.npy", np.array([graph], dtype=object))
    meta = {
        "target": name,
        "source_archive": str(matrix_path),
        "splits": {
            "train": {"num_graphs": 1, "names": [name]},
            "val": {"num_graphs": 1, "names": [name]},
        },
        "graph_meta": graph["meta"],
    }
    (output_root / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
