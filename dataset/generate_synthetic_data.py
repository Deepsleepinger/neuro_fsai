"""Generate synthetic FEM-like training data for Neuro-ILU.

Creates .npy files in the format expected by HeatDatasetMultiSource,
without needing external mesh files or a C++ FEM simulator.

Usage:
    python generate_synthetic_data.py --mesh circle_low_res --diffusivity 100.0 --num-files 100
"""

import os
import argparse
import numpy as np
from scipy.spatial import Delaunay


def generate_mesh_2d(name='circle_low_res', num_nodes=400):
    """Generate a 2D point cloud and triangulation (mimics FEM mesh)."""
    np.random.seed(42)

    if 'circle' in name:
        # Sample points on a disk
        r = np.sqrt(np.random.uniform(0, 1, num_nodes))
        theta = np.random.uniform(0, 2 * np.pi, num_nodes)
        points = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    elif 'eight' in name:
        # Figure-eight shape
        t = np.linspace(0, 2 * np.pi, num_nodes)
        x = np.sin(t)
        y = np.sin(t) * np.cos(t)
        points = np.stack([x, y], axis=1)
        points += np.random.randn(*points.shape) * 0.05
    else:
        # Default: unit square
        points = np.random.uniform(-1, 1, (num_nodes, 2))

    # Delaunay triangulation
    tri = Delaunay(points)
    faces = tri.simplices  # [F, 3]

    # Identify boundary nodes (convex hull)
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    boundary_nodes = set(hull.vertices)

    return points, faces, boundary_nodes


def build_fem_graph(points, faces, boundary_nodes, diffusivity=100.0, dt=0.01):
    """Build FEM graph: edge_index, edge_attr, node_attr, A, rhs."""
    N = points.shape[0]

    # Build edge set from triangular faces
    edge_set = set()
    for f in faces:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = f[i], f[j]
                edge_set.add((int(a), int(b)))
                edge_set.add((int(b), int(a)))
    # Add self-loops
    for i in range(N):
        edge_set.add((i, i))

    edge_list = list(edge_set)
    edge_index = np.array(edge_list).T  # [2, E]
    E = edge_index.shape[1]

    # Edge lengths
    p0 = points[edge_index[0]]
    p1 = points[edge_index[1]]
    edge_len = np.linalg.norm(p0 - p1, axis=1)

    # Synthetic mass (M_e) and stiffness (K_e) contributions per edge
    # M_e ~ edge_len (lumped mass), K_e ~ 1/edge_len (stiffness)
    M_e = edge_len * 0.1
    K_e = diffusivity * dt / (edge_len + 0.01) * 0.5
    edge_attr = np.stack([edge_len, M_e, K_e], axis=1)  # [E, 3]

    # Dirichlet mask: boundary nodes are Dirichlet
    dirichlet_mask = np.zeros(N)
    for bn in boundary_nodes:
        dirichlet_mask[bn] = 1.0

    # Node positions + dirichlet mask (u_value filled in per sample)
    # node_attr layout: [pos_x, pos_y, u_value, dirichlet_mask]
    node_attr = np.zeros((N, 4))
    node_attr[:, :2] = points

    return node_attr, edge_attr, edge_index, dirichlet_mask


def generate_one_sample(node_attr_base, edge_attr, edge_index, dirichlet_mask, diffusivity=100.0, dt=0.01):
    """Generate one (u, u_next, rhs, A_diag, r) sample."""
    N = node_attr_base.shape[0]
    edge_attr = edge_attr.copy()
    edge_index = edge_index.copy()

    # Generate a smooth random field for u
    u = np.random.uniform(-1, 1, N)

    # Smooth u using graph Laplacian smoothing (a few steps)
    for _ in range(10):
        u_new = np.zeros(N)
        counts = np.zeros(N) + 1e-6
        for ei in range(edge_index.shape[1]):
            src, dst = edge_index[0, ei], edge_index[1, ei]
            u_new[dst] += u[src]
            counts[dst] += 1
        u = (u_new / counts) * 0.8 + u * 0.2
    u = (u - u.min()) / (u.max() - u.min() + 1e-8) * 2 - 1

    # Extract A entries per edge
    M_e = edge_attr[:, 1]
    K_e = edge_attr[:, 2]
    A_e = M_e + K_e  # A = M + dt*K for heat equation

    # Build sparse A and compute b = A @ u
    A_dense = np.zeros((N, N))
    for ei in range(edge_index.shape[1]):
        src, dst = edge_index[0, ei], edge_index[1, ei]
        A_dense[dst, src] += A_e[ei]

    # Apply Dirichlet: A[boundary] = identity, b[boundary] = 0
    dirichlet_nodes = np.where(dirichlet_mask > 0)[0]
    for dn in dirichlet_nodes:
        A_dense[dn, :] = 0
        A_dense[:, dn] = 0
        A_dense[dn, dn] = 1.0

    # Ensure symmetry and positive definiteness (heat equation A is SPD)
    A_dense = (A_dense + A_dense.T) / 2
    # Add small epsilon to diagonal for numerical stability
    A_dense += np.eye(N) * 1e-6

    rhs = A_dense @ u
    rhs[dirichlet_nodes] = 0.0

    # Diagonal of A
    diag = np.diag(A_dense)

    # r = residual indicator for edges involving non-dirichlet nodes
    r = np.zeros(edge_index.shape[1])
    for ei in range(edge_index.shape[1]):
        src, dst = edge_index[0, ei], edge_index[1, ei]
        if dirichlet_mask[src] == 0 and dirichlet_mask[dst] == 0:
            r[ei] = A_e[ei] ** 2
    r = r / max(1.0, np.sqrt(np.sum(r)))

    # Build node_attr with u value
    node_attr = node_attr_base.copy()
    node_attr[:, 2] = u
    node_attr[:, 3] = dirichlet_mask

    # u_next = u (steady state assumption for synthetic data)
    u_next = u.copy()

    return {
        'x': node_attr.astype(np.float32),
        'edge_attr': edge_attr.astype(np.float32),
        'edge_index': edge_index,
        'y': rhs.reshape(-1, 1).astype(np.float32),
        'rhs': rhs.reshape(-1, 1).astype(np.float32),
        'u': u.reshape(-1, 1).astype(np.float32),
        'u_next': u_next.reshape(-1, 1).astype(np.float32),
        'diag': diag.reshape(-1, 1).astype(np.float32),
        'r': r.reshape(-1, 1).astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, default='circle_low_res')
    parser.add_argument('--diffusivity', type=float, default=100.0)
    parser.add_argument('--num-nodes', type=int, default=300)
    parser.add_argument('--num-files', type=int, default=100)
    parser.add_argument('--samples-per-file', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='./')
    args = parser.parse_args()

    # Output path: dataset/diffusivity_{d}/{mesh}/
    out_dir = os.path.abspath(os.path.join(args.output_dir,
                           f'diffusivity_{args.diffusivity}',
                           args.mesh))
    os.makedirs(out_dir, exist_ok=True)
    print(f'Output directory: {out_dir}')

    # Generate mesh
    points, faces, boundary_nodes = generate_mesh_2d(
        args.mesh, num_nodes=args.num_nodes
    )
    N = points.shape[0]
    print(f'Mesh: {N} nodes, {faces.shape[0]} faces, '
          f'{len(boundary_nodes)} boundary nodes')

    # Build FEM graph structure
    node_attr_base, edge_attr, edge_index, dirichlet_mask = build_fem_graph(
        points, faces, boundary_nodes, diffusivity=args.diffusivity
    )
    E = edge_index.shape[1]
    print(f'Graph: {N} nodes, {E} edges')

    # Generate data files
    total_samples = 0
    for file_idx in range(args.num_files):
        samples = []
        for _ in range(args.samples_per_file):
            sample = generate_one_sample(
                node_attr_base, edge_attr, edge_index, dirichlet_mask,
                diffusivity=args.diffusivity
            )
            samples.append(sample)
            total_samples += 1

        file_path = os.path.join(out_dir, f'data_{file_idx:04d}.npy')
        np.save(file_path, np.array(samples, dtype=object))
        print(f'  Saved {file_path} ({args.samples_per_file} samples)')

    # Also create a test set with different diffusivity
    test_diff = args.diffusivity * 0.015  # ~1.5 for 100
    test_dir = os.path.abspath(os.path.join(args.output_dir,
                            f'diffusivity_{test_diff:.1f}',
                            f'{args.mesh}_test'))
    os.makedirs(test_dir, exist_ok=True)
    print(f'Test directory: {test_dir}')

    node_attr_test, edge_attr_test, edge_index_test, dirichlet_mask_test = build_fem_graph(
        points, faces, boundary_nodes, diffusivity=test_diff
    )

    for file_idx in range(5):
        samples = []
        for _ in range(args.samples_per_file):
            sample = generate_one_sample(
                node_attr_test, edge_attr_test, edge_index_test, dirichlet_mask_test,
                diffusivity=test_diff
            )
            samples.append(sample)

        file_path = os.path.join(test_dir, f'data_{file_idx:04d}.npy')
        np.save(file_path, np.array(samples, dtype=object))
        print(f'  Saved {file_path} ({args.samples_per_file} samples)')

    print(f'\nDone. Generated {total_samples} training + {5 * args.samples_per_file} test samples.')


if __name__ == '__main__':
    main()
