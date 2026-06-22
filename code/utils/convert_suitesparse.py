"""Convert SuiteSparse .mtx matrices to Neuro-ILU training data format.

Each matrix A is converted to a graph:
  - Nodes: rows of A, with structural features (degree, diagonal, row-sum)
  - Edges: non-zero entries A[i,j] → directed edge (j→i) with value as feature
  - Target: randomly sampled x and b = A @ x

Output: .npy file compatible with SyntheticDataset loader.

Usage:
    python convert_suitesparse.py \
        --input-dir /mnt/h/suitesparse \
        --output-dir /mnt/h/neuro_ilu/data \
        --num-matrices 200 --min-size 100 --max-size 2000
"""

import os
import sys
import argparse
import tarfile
import io
import hashlib
import numpy as np
from scipy.io import mmread
from scipy.sparse import issparse, triu, tril, eye, csr_matrix
from collections import defaultdict

from utils.topology_expansion import expand_sparse_topology_scipy


def read_mtx_from_tar(tar_path):
    """Read a .mtx matrix from a .tar.gz file."""
    with tarfile.open(tar_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.mtx'):
                f = tar.extractfile(member)
                if f:
                    return mmread(io.BytesIO(f.read()))
    return None


def is_non_symmetric(tar_path):
    """Check if a SuiteSparse archive contains a non-symmetric matrix."""
    with tarfile.open(tar_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.mtx'):
                f = tar.extractfile(member)
                if f:
                    header = f.readline().decode('utf-8').strip()
                    return 'general' in header.lower()
    return False


def canonicalize_sparse_matrix(A):
    """Return a CSR matrix with duplicates summed and explicit zeros removed."""
    if not issparse(A):
        A = csr_matrix(A)
    else:
        A = A.tocoo()
        A.sum_duplicates()
        A.eliminate_zeros()
        A = A.tocsr()
    return A


def values_on_pattern(A, row, col):
    """Return A[row, col] for a candidate pattern, zero for structural fill-ins."""
    A = canonicalize_sparse_matrix(A)
    A_coo = A.tocoo()
    n = A.shape[1]
    orig_keys = A_coo.row.astype(np.int64) * n + A_coo.col.astype(np.int64)
    order = np.argsort(orig_keys)
    orig_keys = orig_keys[order]
    orig_values = A_coo.data.astype(np.float64, copy=False)[order]

    query_keys = row.astype(np.int64, copy=False) * n + col.astype(np.int64, copy=False)
    if orig_keys.size == 0:
        return np.zeros(row.shape[0], dtype=np.float64)
    pos = np.searchsorted(orig_keys, query_keys)
    clipped_pos = pos.clip(max=orig_keys.size - 1)
    matched = (pos < orig_keys.size) & (orig_keys[clipped_pos] == query_keys)
    values = np.zeros(row.shape[0], dtype=np.float64)
    if matched.any():
        values[matched] = orig_values[pos[matched]]
    return values


def matrix_to_graph(A, name='unknown', topology_hop=1,
                    max_topology_edges=None, max_topology_ratio=None):
    """Convert sparse matrix A to Neuro-ILU graph format."""
    A = canonicalize_sparse_matrix(A)
    N = A.shape[0]

    # ---- Edge construction from candidate topology ----
    # The matrix values remain those of A. Extra 2-hop candidate edges get
    # value 0.0 and are only allowed locations for learned FSAI factors.
    topology, topology_expanded = expand_sparse_topology_scipy(
        A,
        topology_hop=topology_hop,
        max_topology_edges=max_topology_edges,
        max_topology_ratio=max_topology_ratio)
    topology_coo = topology.tocoo()
    row = topology_coo.row.astype(np.int64, copy=False)
    col = topology_coo.col.astype(np.int64, copy=False)
    values = values_on_pattern(A, row, col)
    original_nnz = A.nnz

    # edge_index: [2, E] with (col → row) meaning A[row, col]
    edge_index = np.stack([col, row], axis=0).astype(np.int64)
    E = edge_index.shape[1]

    # ---- Node features: structural properties ----
    # [pos_x, pos_y, diag_val, interior_mask]
    node_attr = np.zeros((N, 4), dtype=np.float32)

    # pos_x: normalized row index
    node_attr[:, 0] = np.arange(N) / max(1, N - 1) * 2 - 1

    # pos_y: normalized out-degree (number of non-zeros per row)
    row_degrees = np.bincount(row, minlength=N).astype(np.float32)
    max_deg = max(1, row_degrees.max())
    node_attr[:, 1] = row_degrees / max_deg * 2 - 1

    # diag_val placeholder in node_attr[:, 2] — filled per sample
    # interior_mask in node_attr[:, 3] — all interior for generic matrices
    node_attr[:, 3] = 1.0  # mark all as interior (no Dirichlet boundaries)

    # ---- Edge features: [edge_len, value, structural_flag] ----
    edge_attr = np.zeros((E, 3), dtype=np.float32)

    # edge_len: normalized index distance |row - col| / N
    idx_dist = np.abs(row - col).astype(np.float32)
    edge_attr[:, 0] = idx_dist / N * 2 - 1

    # value: the matrix entry
    edge_attr[:, 1] = values.astype(np.float32)

    # structural_flag: 1.0 for self-loop (diagonal), 0.0 otherwise
    edge_attr[:, 2] = (row == col).astype(np.float32)

    # ---- Diagonal of A ----
    diag = np.array(A.diagonal(), dtype=np.float32)

    # ---- Generate random solution ----
    seed = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, N).astype(np.float64)

    # Compute rhs = A @ u
    rhs = A.dot(u)
    rhs = np.array(rhs, dtype=np.float32).flatten()
    u = u.astype(np.float32)

    # ---- Residual indicator r (for edges) ----
    r = np.where(row != col, np.abs(values), 0.0).astype(np.float32)
    r_max = max(1e-8, float(np.max(np.abs(r))) if r.size else 0.0)
    r_scaled = (r / r_max).astype(np.float32)
    r_norm = max(1e-8, float(np.linalg.norm(r_scaled.astype(np.float64))))
    r = r_scaled / r_norm

    # Fill node_attr[2] with u values + node_attr
    node_attr[:, 2] = u

    # u_next = u (for static problem, Ax=b)
    u_next = u.copy()

    return {
        'x': node_attr,
        'edge_attr': edge_attr,
        'edge_index': edge_index,
        'y': rhs.reshape(-1, 1),
        'rhs': rhs.reshape(-1, 1),
        'u': u.reshape(-1, 1),
        'u_next': u_next.reshape(-1, 1),
        'diag': diag.reshape(-1, 1),
        'r': r.reshape(-1, 1),
        'meta': {
            'name': name,
            'N': N,
            'E': E,
            'nnz': int(original_nnz),
            'topology_hop': int(topology_hop),
            'topology_E': int(E),
            'topology_expanded': bool(topology_expanded),
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to SuiteSparse .tar.gz files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for .npy data files')
    parser.add_argument('--num-matrices', type=int, default=200,
                        help='Number of matrices to convert')
    parser.add_argument('--min-size', type=int, default=100,
                        help='Minimum matrix size')
    parser.add_argument('--max-size', type=int, default=2000,
                        help='Maximum matrix size')
    parser.add_argument('--samples-per-matrix', type=int, default=1,
                        help='Number of random RHS per matrix')
    args = parser.parse_args()

    # Find all .tar.gz files
    all_tars = sorted([
        f for f in os.listdir(args.input_dir) if f.endswith('.tar.gz')
    ])
    print(f'Found {len(all_tars)} archives in {args.input_dir}')

    # Filter: non-symmetric only
    non_sym_tars = []
    for t in all_tars:
        tarpath = os.path.join(args.input_dir, t)
        if is_non_symmetric(tarpath):
            non_sym_tars.append(tarpath)
    print(f'Non-symmetric: {len(non_sym_tars)}')

    # Create output directories
    train_dir = os.path.join(args.output_dir, 'train')
    test_dir = os.path.join(args.output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Process matrices
    converted = 0
    train_graphs = []
    test_graphs = []
    skipped = 0

    for tarpath in non_sym_tars:
        if converted >= args.num_matrices:
            break

        name = os.path.basename(tarpath).replace('.tar.gz', '')

        try:
            A = read_mtx_from_tar(tarpath)
        except Exception as e:
            print(f'  Skip {name}: read error ({e})')
            skipped += 1
            continue

        if A is None:
            skipped += 1
            continue

        N = A.shape[0]
        if N < args.min_size or N > args.max_size:
            skipped += 1
            continue

        nnz = A.nnz if issparse(A) else np.count_nonzero(A)
        print(f'[{converted+1}/{args.num_matrices}] {name}: N={N}, nnz={nnz}')

        try:
            graph_data = matrix_to_graph(A, name=name)
        except Exception as e:
            print(f'  Skip {name}: conversion error ({e})')
            skipped += 1
            continue

        # Generate additional samples with different random RHS
        samples = [graph_data]
        for s in range(1, args.samples_per_matrix):
            sample = matrix_to_graph(A, name=f'{name}_s{s}')
            samples.append(sample)

        # Split: 80% train, 20% test
        if converted % 5 == 0:
            test_graphs.extend(samples)
        else:
            train_graphs.extend(samples)

        converted += 1

    print(f'\nConverted: {converted} matrices, skipped: {skipped}')
    print(f'Train graphs: {len(train_graphs)}')
    print(f'Test graphs: {len(test_graphs)}')

    # Save as .npy files (chunked)
    def save_chunked(graphs, out_dir, prefix, chunk_size=100):
        for start in range(0, len(graphs), chunk_size):
            end = min(start + chunk_size, len(graphs))
            chunk = graphs[start:end]
            path = os.path.join(out_dir, f'{prefix}_{start//chunk_size:04d}.npy')
            np.save(path, np.array(chunk, dtype=object))
            print(f'  Saved {path} ({len(chunk)} graphs)')

    save_chunked(train_graphs, train_dir, 'train')
    save_chunked(test_graphs, test_dir, 'test')

    # Save metadata
    meta = {
        'num_train': len(train_graphs),
        'num_test': len(test_graphs),
        'min_N': min(g['meta']['N'] for gs in [train_graphs, test_graphs] for g in gs),
        'max_N': max(g['meta']['N'] for gs in [train_graphs, test_graphs] for g in gs),
        'names': list(set(g['meta']['name'] for g in train_graphs + test_graphs)),
    }
    np.save(os.path.join(args.output_dir, 'meta.npy'), meta)
    print(f'\nMetadata saved. N range: [{meta["min_N"]}, {meta["max_N"]}]')
    print('Done.')


if __name__ == '__main__':
    main()
