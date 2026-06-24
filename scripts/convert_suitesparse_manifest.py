import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import identity
from scipy.sparse.csgraph import reverse_cuthill_mckee

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.convert_suitesparse import matrix_to_graph, read_mtx_from_tar


class ChunkedWriter:
    def __init__(self, out_dir, prefix, chunk_size):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.chunk_size = chunk_size
        self.buffer = []
        self.chunk_index = 0
        self.total_graphs = 0

    def add(self, graph):
        self.buffer.append(graph)
        self.total_graphs += 1
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        out_path = self.out_dir / f"{self.prefix}_{self.chunk_index:04d}.npy"
        np.save(out_path, np.array(self.buffer, dtype=object))
        print(f"Saved {out_path} ({len(self.buffer)} graphs)")
        self.buffer = []
        self.chunk_index += 1


def split_summary():
    return {
        "num_graphs": 0,
        "num_matrices": 0,
        "min_n": None,
        "max_n": None,
        "min_edges": None,
        "max_edges": None,
        "total_edges": 0,
        "topology_expanded": 0,
        "topology_fallback": 0,
        "names": [],
    }


def update_summary(summary, graph):
    graph_name = graph["meta"]["name"]
    nrows = graph["meta"]["N"]
    edges = graph["meta"]["E"]
    summary["num_graphs"] += 1
    summary["min_n"] = nrows if summary["min_n"] is None else min(summary["min_n"], nrows)
    summary["max_n"] = nrows if summary["max_n"] is None else max(summary["max_n"], nrows)
    summary["min_edges"] = edges if summary["min_edges"] is None else min(summary["min_edges"], edges)
    summary["max_edges"] = edges if summary["max_edges"] is None else max(summary["max_edges"], edges)
    summary["total_edges"] += edges
    if graph["meta"].get("topology_hop") == 2:
        if graph["meta"].get("topology_expanded"):
            summary["topology_expanded"] += 1
        else:
            summary["topology_fallback"] += 1
    summary["names"].append(graph_name)


def parse_augmentations(spec):
    augmentations = []
    for item in spec.split(","):
        item = item.strip().lower()
        if item:
            augmentations.append(item)
    return augmentations or ["identity"]


def permute_matrix(A, mode, seed):
    if mode == "identity":
        return A
    if mode == "rcm":
        perm = reverse_cuthill_mckee(A.tocsr(), symmetric_mode=False)
    elif mode == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(A.shape[0])
    else:
        raise ValueError(f"Unknown augmentation: {mode}")

    P = identity(A.shape[0], format="csr", dtype=A.dtype)[perm, :]
    return P @ A @ P.T


def convert_split(rows, split_name, output_root, samples_per_matrix, chunk_size,
                  augmentations, augment_val_test, topology_hop,
                  max_topology_edges, max_topology_ratio):
    writer = ChunkedWriter(Path(output_root) / split_name, split_name, chunk_size)
    summary = split_summary()
    summary["num_matrices"] = len(rows)

    for row in rows:
        archive_path = row.get("copied_to") or row["path"]
        if not Path(archive_path).exists() and row.get("path"):
            archive_path = row["path"]
        matrix = read_mtx_from_tar(archive_path)
        if matrix is None:
            raise RuntimeError(f"Failed to read matrix from {archive_path}")

        base_name = row["name"]
        split_augmentations = augmentations
        if split_name != "train" and not augment_val_test:
            split_augmentations = ["identity"]

        for aug_idx, augmentation in enumerate(split_augmentations):
            seed = abs(hash((base_name, augmentation))) % (2 ** 32)
            aug_matrix = permute_matrix(matrix, augmentation, seed)
            aug_suffix = "" if augmentation == "identity" else f"_{augmentation}"
            graph_name = f"{base_name}{aug_suffix}"
            graph = matrix_to_graph(
                aug_matrix,
                name=graph_name,
                topology_hop=topology_hop,
                max_topology_edges=max_topology_edges,
                max_topology_ratio=max_topology_ratio)
            writer.add(graph)
            update_summary(summary, graph)

            for sample_idx in range(1, samples_per_matrix):
                sample_name = f"{graph_name}_s{sample_idx}"
                sample_graph = matrix_to_graph(
                    aug_matrix,
                    name=sample_name,
                    topology_hop=topology_hop,
                    max_topology_edges=max_topology_edges,
                    max_topology_ratio=max_topology_ratio)
                writer.add(sample_graph)
                update_summary(summary, sample_graph)

    writer.flush()
    summary["num_chunks"] = writer.chunk_index
    if summary["num_graphs"]:
        summary["avg_edges"] = summary["total_edges"] / summary["num_graphs"]
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument(
        "--output-root",
        type=str,
        default="prepared/suitesparse_balanced_v1/data",
    )
    parser.add_argument("--samples-per-matrix", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--augmentations",
        type=str,
        default="identity",
        help="comma-separated augmentations: identity,rcm,random")
    parser.add_argument(
        "--augment-val-test",
        action="store_true",
        help="apply augmentations to validation/test splits too")
    parser.add_argument(
        "--topology-hop",
        type=int,
        default=1,
        choices=[1, 2],
        help="candidate FSAI topology: 1=A pattern, 2=union of A and A^2 patterns")
    parser.add_argument(
        "--max-topology-edges",
        type=int,
        default=0,
        help="fallback to 1-hop when expanded topology exceeds this edge count; <=0 disables")
    parser.add_argument(
        "--max-topology-ratio",
        type=float,
        default=0.0,
        help="fallback to 1-hop when expanded/original edge ratio exceeds this value; <=0 disables")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_summary = convert_split(
        manifest["splits"]["train"],
        "train",
        output_root,
        args.samples_per_matrix,
        args.chunk_size,
        parse_augmentations(args.augmentations),
        args.augment_val_test,
        args.topology_hop,
        args.max_topology_edges,
        args.max_topology_ratio,
    )
    val_summary = convert_split(
        manifest["splits"]["val"],
        "val",
        output_root,
        args.samples_per_matrix,
        args.chunk_size,
        parse_augmentations(args.augmentations),
        args.augment_val_test,
        args.topology_hop,
        args.max_topology_edges,
        args.max_topology_ratio,
    )
    test_summary = None
    if "test" in manifest["splits"]:
        test_summary = convert_split(
            manifest["splits"]["test"],
            "test",
            output_root,
            args.samples_per_matrix,
            args.chunk_size,
            parse_augmentations(args.augmentations),
            args.augment_val_test,
            args.topology_hop,
            args.max_topology_edges,
            args.max_topology_ratio,
        )

    meta = {
        "manifest": str(Path(args.manifest).resolve()),
        "samples_per_matrix": args.samples_per_matrix,
        "chunk_size": args.chunk_size,
        "augmentations": parse_augmentations(args.augmentations),
        "augment_val_test": args.augment_val_test,
        "topology_hop": args.topology_hop,
        "max_topology_edges": args.max_topology_edges,
        "max_topology_ratio": args.max_topology_ratio,
        "splits": {
            "train": train_summary,
            "val": val_summary,
        },
    }
    if test_summary is not None:
        meta["splits"]["test"] = test_summary

    meta_path = output_root / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["splits"], indent=2))
    print(f"Metadata written to {meta_path}")


if __name__ == "__main__":
    main()
