from cgitb import enable
from marshal import load
import sys
import math
import time
import torch
import numpy as np
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import DataLoader
from scipy.ndimage import gaussian_filter
from scipy import interpolate
from tqdm import tqdm

from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve_triangular, aslinearoperator

from sys import path
path.append('../../python/example/phys_gnn/')
path.append('../../python/py_phys_sim/')
path.append('../../python/')

torch.set_num_threads(64)

def ic( A ):
    mat = np.copy( A )
    n = mat.shape[1]
    
    for k in range( n ):
        mat[k,k] = math.sqrt( mat[k,k] )
        for i in range(k+1, n):
            if mat[i,k] != 0:
                mat[i,k] = mat[i,k] / mat[k,k]
        for j in range(k+1, n):
            for i in range(j, n):
                if mat[i,j] != 0:
                    mat[i,j] = mat[i,j] - mat[i,k] * mat[j,k]
    for i in range(n):
        for j in range(i+1, n):
            mat[i,j] = 0
    
    return mat


# implementing IC based on TAO's c++ code
def ic_torch_st( A , device='cuda:0'):
    A = A.to(device)
    ic = A.clone()
    A_copy = A.clone()
    n = A.shape[0]
    for j in tqdm(range(n)): 
        ic[j,j] = torch.sqrt(ic[j,j])
        i_ = ic[j+1:,j].nonzero()
        assert i_.shape[-1] == 1
        i_ = i_.reshape(-1)
        i_ += (j+1)
        for i in i_:
            ic[i, j]  = ic[i, j] - (ic[i, :]*ic[j, :]).reshape(-1)[:j].sum()
            ic[i, j] = ic[i, j] / ic[j, j]

    ic = torch.tril(ic)

    return ic


# IC implementation with tensorized parallel speedup: --> should use this for comparison (this is faster)
def ic_torch_optimize_st( A , device='cuda:0'):
    A = A.to(device)
    ic = A.clone()
    A_copy = A.clone()
    n = A.shape[0]
    
    for j in tqdm(range(n)): 
        ic[j,j] = torch.sqrt(ic[j,j])
        i_ = ic[j+1:,j].nonzero()
        if i_.shape[0] > 0:
            assert i_.shape[-1] == 1
            i_ = i_.reshape(-1)
            i_ += (j+1)
            inner_sum = (ic[i_, :]*ic[j, :])
            inner_sum = inner_sum[:, :j].sum(dim=-1)            
            ic[i_, j]  = ic[i_, j] - inner_sum
            ic[i_, j] = ic[i_, j] / ic[j, j]

    ic = torch.tril(ic)

    return ic


def ic_np( A ):
    A = A.astype(np.float128)
    ic = A.copy()
    print(A)
    n = A.shape[0]
    for j in range(n): 
        assert ic[j, j] > 0, f"{j} {ic[j, j]} "

        ic[j,j] = np.sqrt(ic[j,j])
        # print(A[j, j])
        i_ = (ic[j+1:,j].nonzero())[0]
        i_ = i_.reshape(-1)
        i_ += (j+1)
        for i in i_:
            print((ic[i, :].reshape(-1)).shape)
            inner_sum = np.multiply(ic[i, :].reshape(-1), ic[j, :].reshape(-1)).reshape(-1)[:j].sum()
            # if ic[i, j] < inner_sum: print(i, j, ic[i, j], inner_sum)
            ic[i, j]  = ic[i, j] - inner_sum
            assert ic[j, j] > 0.0, f"diagonal negative"
            ic[i, j] = ic[i, j] / ic[j, j]
            ic[i, i] = ic[i, i] - (ic[i, j] ** 2)
            assert np.all(np.diag(A) > 0.0), f" after: number at {i} {j} is {A[i, i]}, {ic[i, j]} {ic[j, j]} {inner_sum} "

    ic = torch.tril(ic)

    return ic




def ic(A):
    n = A.shape[0]
    A_copy = A.clone()
    ic = A.copy()
    for j in range(n):
        ic[j, j] = np.sqrt(A[j,j])
        for i in range(j+1, n):
            for k in range(j):
                ic[i, j] = ic[i, j] - ic[i, k] * ic[j, k]
            ic[i, j] = ic[i, j] / ic[j, j]
            A[i, i] = A[i, i] - ic[i, j] * ic[i, j]
    
    ic = np.tril(ic)
    return ic


def ic_torch_old( A , device='cuda:0'):
    A = A.to(device)
    ic = A.copy()
    n = A.shape[0]
    for k in range(n): 
        ic[k,k] = torch.sqrt(A[k,k])
        i_ = ic[k+1:,k].nonzero() 
        assert i_.shape[-1] == 1
        i_ = i_.view(-1)
        if i_.shape[0] > 0:
            i_ = i_ + (k+1)
            ic[i_,k] = ic[i_,k]/ic[k,k]
            for j in i_:
                i2_ = ic[j:n,j].nonzero()
                assert i2_.shape[-1] == 1
                i2_ = i2_.view(-1)
                # if len(i2_) > 0:
                if i2_.shape[0] < 0:
                    i2_ = i2_ + j
                    factor = ic[j,k]
                    ic[i2_, j]  = ic[i2_, j] - ic[i2_,k]*factor
            # for j in 
    
    ic = torch.tril(ic)

    return ic


# jacobi based on C++ code
def jacobi(A):
    mat = np.zeros(A.shape)
    for i in range(A.shape[0]):
        mat[i, i] = 1.0/A[i, i]
    return mat


# faster jacobi (tensorized) --> should use this for comparison this is faster
def jacobi_torch(A, device='cuda:0'):
    mat = torch.eye(A.shape[0], device=device)
    mat = mat * (1.0 / torch.diagonal(A).to(device) )
    return mat



def gs_torch(A, device='cuda:0'):
    L = torch.tril(A)
    U = torch.triu(A, diagonal=1)
    return L, U


def cholesky_eigen(A):
    ''' this is complete cholesky decomposition
    '''
    import eigenpy
    L = eigenpy.LLT(A.numpy())
    L = L.matrixL()
    L = np.array(L)
    return L

def cg( A, b, options, model=None, device='cuda:0'):
    A = A.to(device)
    b = b.to(device)
    start_cg_time = time.time()
    rel_tol = options['abs_tol']
    abs_tol = options['rel_tol']
    rel_tols = [1e-4, 1e-6, 1e-8, 1e-10]
    abs_tols = [ 1e-4, 1e-6, 1e-8, 1e-10]
    max_iter = options['max_iter']
    first_flag = True
    second_flag = True
    preconditioner = options['precondition_matrix']

    if 'x' not in options:
        x = torch.zeros((A.shape[0], 1), device=A.device).double()
    else:
        x = options['x']

    r = A @ x - b
    y = torch.mm( preconditioner, r )
    p = -y
    convergent_iterations = {}
    for i in range(max_iter):
       
        Ap       = torch.mm( A , p )
        alpha    = torch.mm(r.T, y)/torch.mm( p.T, Ap )
        x        = x + alpha * p
        r_next   = r + alpha * Ap


        if first_flag and torch.abs(r_next).max() <= torch.abs(b).max() * rel_tols[0] + abs_tols[0]:
            end_cg_time = time.time()
            print(f'1e-4 Pcg Converged in {i} steps time ')
            convergent_iterations['1e-4'] = i
            first_flag = False
        if second_flag and torch.abs(r_next).max() <= torch.abs(b).max() * rel_tols[1] + abs_tols[1]:
            end_cg_time = time.time()
            print(f'1e-6 Pcg Converged in {i} steps time ')
            convergent_iterations['1e-6'] = i
            second_flag = False
        if  torch.abs(r_next).max() <= torch.abs(b).max() * rel_tols[2] + abs_tols[2]:
            end_cg_time = time.time()
            print(f'1e-8 Pcg Converged in {i} steps time {end_cg_time - start_cg_time}')
            convergent_iterations['1e-8'] = i
            return i, convergent_iterations
        
        y_next   = torch.mm( preconditioner, r_next )
        beta     = torch.mm(y_next.T, (r_next - r))/ r.T.mm(y) # Polak-Ribiere
        p        = -y_next + beta * p
        y = y_next
        r = r_next

    if i >= (max_iter-1):
        print('Convergence failed.')
        
    end_cg_time = time.time()


    return 1000000, convergent_iterations



def cg_np(A, b, options):
    ''' single threaded numpy conjugate gradient that supports sparse triangular solver using the naive scipy sparse implementation
    '''
    A = A.cpu().numpy()
    b = b.cpu().numpy()
    start_cg_time = time.time()
    solve_triag = options['sptriangular']
    solve_sparse = options['solve_sparse']

    residual = []
    abs_tols = options['abs_tols']
    rel_tols = options['rel_tols']
    tols_dict = options['tol_dict']
    max_iter = options['max_iter']
    flags = [True for x in rel_tols]
    max_iter = options['max_iter']
    preconditioner = options['precondition_matrix']

    if 'x' not in options:
        x = np.zeros((A.shape[0], 1))
    else:
        x = options['x'].cpu().numpy()
        

    r =  A.dot(x) - b
    if solve_triag:
        r_new = r.reshape(-1, 1)
        r_new = np.concatenate([r_new, np.zeros_like(r_new)], axis=-1)
        y0 = spsolve_triangular(preconditioner.transpose(), r_new, lower=False)
        y = spsolve_triangular(preconditioner, y0, lower=True)
        y = y[:, 0].reshape(-1,1)
    elif solve_sparse:
        r_new = r
        y = preconditioner.matvec(r_new)
        y = y.reshape(-1)
    else:
        y = np.dot( preconditioner, r )
    p = -y

    convergent_iterations = {}
    convergent_time = {}
    for i in range(max_iter):
        Ap       = np.dot( A , p )
        alpha    = np.dot(r.T, y)/np.dot( p.T, Ap )
        x        = x + alpha * p
        r_next   = r.reshape(-1) + alpha.reshape(-1) * Ap.reshape(-1)

        r = r.reshape(-1)
        y = y.reshape(-1)
        
        for j in range(len(rel_tols)-1):
            if flags[j] and np.abs(r_next).max() <= np.abs(b).max() * rel_tols[j] + abs_tols[j]:
                # delta_time =  time.time() - iter_start_time
                delta_time =  time.time() - start_cg_time
                iter_start_time = time.time()
                print(f'{tols_dict[j]} Pcg Converged in {i} steps ')
                convergent_iterations[tols_dict[j]] = i
                convergent_time[tols_dict[j]] = delta_time
                flags[j] = False
                
        if  np.abs(r_next).max() <= np.abs(b).max() * rel_tols[-1] + abs_tols[-1]:
            delta_time =  time.time() - start_cg_time
            iter_start_time = time.time()
            print(f'{tols_dict[-1]} Pcg Converged in {i} steps {iter_start_time - start_cg_time}')
            convergent_iterations[tols_dict[-1]] = i
            convergent_time[tols_dict[-1]] = delta_time
            return i, convergent_iterations, convergent_time, residual
        
        if solve_triag:
            r_next_new = r_next.reshape(-1, 1)
            r_next_new = np.concatenate([r_next, np.zeros_like(r_next_new)], axis=-1)
            y0 = spsolve_triangular(preconditioner.transpose(), r_next_new, lower=False)
            y_next = spsolve_triangular(preconditioner, y0, lower=True)
            y_next = y_next[:, 0].reshape(-1, 1)
        elif solve_sparse:
            r_new = r_next.reshape(-1)#.reshape(-1, 1)
            y_next = preconditioner.matvec(r_new)
            y_next = y_next#.reshape(-1,1)
        else:
            y_next = np.dot( preconditioner, r_next )
    
        beta     = np.dot(y_next.T, (r_next - r))/ np.dot(r.T, y) # Polak-Ribiere
        p        = -y_next + beta * p
        y = y_next
        r = r_next

    if i >= (max_iter-1):
        print('Convergence failed.')
        
    end_cg_time = time.time()


    return 1000000, convergent_iterations, convergent_time, residual




def cg_torch(A, b, options, model=None, device='cuda:0', num_threads=64, plot=False):
    ''' pytorch implementation of conjugate gradient that supports gpu/cpu and multithreading 
        use max as stopping criteria
    '''
    torch.set_num_threads(num_threads)

    start_cg_time = time.time()
    iter_start_time = time.time()

    abs_tols = options['abs_tols']
    rel_tols = options['rel_tols']
    tol_dict = options['tol_dict']
    max_iter = options['max_iter']
    flags = [True for x in rel_tols]
    preconditioner = options['precondition_matrix']
    x = options['x']

    residual=[]

    A = A.to(device).float()
    b = b.to(device).float()
    x = x.to(device).float()
    preconditioner = preconditioner.to(device).float()
    

    r = A @ x - b
    y = torch.mm( preconditioner, r )
    p = -y
    convergent_iterations = {}
    convergent_time = {}
    for i in range(max_iter):
        Ap       = torch.mm( A , p )
        alpha    = torch.mm(r.T, y)/torch.mm( p.T, Ap )
        x        = x + alpha * p
        r_next   = r + alpha * Ap

        if plot:
            res = torch.abs(r_next).max()
            delta_time =  time.time() - iter_start_time
            iter_start_time = time.time()
            residual.append((res.item(), delta_time))

        for j in range(len(rel_tols)-1):
            if flags[j] and torch.abs(r_next).max() <= torch.abs(b).max() * rel_tols[j] + abs_tols[j]:
                delta_time =  time.time() - start_cg_time
                iter_start_time = time.time()
                print(f'{tol_dict[j]} Pcg Converged in {i} steps ')
                convergent_iterations[tol_dict[j]] = i
                convergent_time[tol_dict[j]] = delta_time
                flags[j] = False
                
        if  torch.abs(r_next).max() <= torch.abs(b).max() * rel_tols[-1] + abs_tols[-1]:
            delta_time =  time.time() - start_cg_time
            iter_start_time = time.time()
            print(f'{tol_dict[-1]} Pcg Converged in {i} steps {iter_start_time - start_cg_time}')
            convergent_iterations[tol_dict[-1]] = i
            convergent_time[tol_dict[-1]] = delta_time
            return i, convergent_iterations, convergent_time, residual
        
        y_next   = torch.mm( preconditioner, r_next)
        beta     = torch.mm(y_next.T, (r_next - r))/ r.T.mm(y) # Polak-Ribiere
        p        = -y_next + beta * p
        y = y_next
        r = r_next

    if i >= (max_iter-1):
        print('Convergence failed.')
        
    end_cg_time = time.time()


    return 1000000, convergent_iterations, convergent_time, residual


def cg_torch_mean(A, b, options, model=None, device='cuda:0', num_threads=64, plot=False):
    ''' pytorch implementation of conjugate gradient that supports gpu/cpu and multithreading 
        use mean as stopping criteria
    '''
    torch.set_num_threads(num_threads)

    start_cg_time = time.time()
    iter_start_time = time.time()

    abs_tols = options['abs_tols']
    rel_tols = options['rel_tols']
    tol_dict = options['tol_dict']
    max_iter = options['max_iter']
    flags = [True for x in rel_tols]
    preconditioner = options['precondition_matrix']
    x = options['x']

    residual=[]

    A = A.to(device)
    b = b.to(device)
    x = x.to(device)
    preconditioner = preconditioner.to(device)

    r = A @ x - b
    y = torch.mm( preconditioner, r )
    p = -y

    convergent_iterations = {}
    convergent_time = {}
    for i in range(max_iter):

        Ap       = torch.mm( A , p )
        alpha    = torch.mm(r.T, y)/torch.mm( p.T, Ap )
        x        = x + alpha * p
        r_next   = r + alpha * Ap

        if plot:
            res = torch.abs(r_next).max()
            delta_time =  time.time() - iter_start_time
            iter_start_time = time.time()
            residual.append((res.item(), delta_time))

        for j in range(len(rel_tols)-1):
            if flags[j] and torch.abs(r_next).mean() <= torch.abs(b).max() * rel_tols[j] + abs_tols[j]:
                delta_time =  time.time() - start_cg_time
                iter_start_time = time.time()
                print(f'{tol_dict[j]} Pcg Converged in {i} steps ')
                convergent_iterations[tol_dict[j]] = i
                convergent_time[tol_dict[j]] = delta_time
                flags[j] = False
                
        if  torch.abs(r_next).mean() <= torch.abs(b).max() * rel_tols[-1] + abs_tols[-1]:
            delta_time =  time.time() - start_cg_time
            iter_start_time = time.time()
            print(f'{tol_dict[-1]} Pcg Converged in {i} steps {iter_start_time - start_cg_time}')
            convergent_iterations[tol_dict[-1]] = i
            convergent_time[tol_dict[-1]] = delta_time
            return i, convergent_iterations, convergent_time, residual

        y_next   = torch.mm( preconditioner, r_next)
        beta     = torch.mm(y_next.T, (r_next - r))/ r.T.mm(y) # Polak-Ribiere
        p        = -y_next + beta * p
        y = y_next
        r = r_next

    if i >= (max_iter-1):
        print('Convergence failed.')

    end_cg_time = time.time()


    return 1000000, convergent_iterations, convergent_time, residual


# ============================================================
# BiCGSTAB — for non-symmetric systems with Neuro-ILU preconditioner
# ============================================================

def bicgstab_torch(A, b, L_dense, U_dense, options, device='cuda:0'):
    """Preconditioned BiCGSTAB for non-symmetric A x = b.

    Uses M^{-1} = U^{-1} L^{-1} (left preconditioning) applied via
    two triangular solves per iteration.
    """
    N = A.shape[0]
    abs_tol = options.get('abs_tol', 1e-9)
    rel_tol = options.get('rel_tol', 0.0)
    max_iter = options.get('max_iter', 2000)

    A = A.to(device).double()
    b = b.to(device).double()
    L_dense = L_dense.to(device).double()
    U_dense = U_dense.to(device).double()

    x = torch.zeros(N, 1, device=device, dtype=torch.float64)

    r = b - A @ x
    r_hat = r.clone()

    rho_prev = alpha = omega = 1.0
    v = torch.zeros(N, 1, device=device, dtype=torch.float64)
    p = torch.zeros(N, 1, device=device, dtype=torch.float64)

    b_norm = torch.abs(b).max()

    for i in range(max_iter):
        rho = (r_hat.T @ r).squeeze()

        if abs(rho) < 1e-30:
            return max_iter, {}

        if i == 0:
            p = r.clone()
        else:
            beta = (rho / rho_prev) * (alpha / omega)
            p = r + beta * (p - omega * v)

        # Apply preconditioner: p_hat = M^{-1} p = U^{-1} L^{-1} p
        w = torch.linalg.solve_triangular(L_dense, p, upper=False, unitriangular=True)
        p_hat = torch.linalg.solve_triangular(U_dense, w, upper=True, unitriangular=False)

        v = A @ p_hat
        r_hat_dot_v = (r_hat.T @ v).squeeze()

        if abs(r_hat_dot_v) < 1e-30:
            return max_iter, {}

        alpha = rho / r_hat_dot_v
        s = r - alpha * v

        s_norm = torch.abs(s).max()
        if s_norm <= b_norm * rel_tol + abs_tol:
            x = x + alpha * p_hat
            return i + 1, {str(rel_tol): i + 1}

        # Apply preconditioner to s
        w_s = torch.linalg.solve_triangular(L_dense, s, upper=False, unitriangular=True)
        s_hat = torch.linalg.solve_triangular(U_dense, w_s, upper=True, unitriangular=False)

        t = A @ s_hat
        t_dot_s = (t.T @ s).squeeze()
        t_dot_t = (t.T @ t).squeeze()

        if abs(t_dot_t) < 1e-30:
            omega = 0.0
        else:
            omega = t_dot_s / t_dot_t

        x = x + alpha * p_hat + omega * s_hat
        r = s - omega * t

        r_norm = torch.abs(r).max()
        if r_norm <= b_norm * rel_tol + abs_tol:
            return i + 1, {str(rel_tol): i + 1}

        if abs(omega) < 1e-30:
            return max_iter, {}

        rho_prev = rho

    return max_iter, {}


def edge_to_csr(edge_index, values, N):
    """Convert sparse edge-index data to scipy CSR format."""
    row = edge_index[0].detach().cpu().numpy()
    col = edge_index[1].detach().cpu().numpy()
    vals = values.detach().cpu().numpy().astype(np.float64)
    return csr_matrix((vals, (row, col)), shape=(N, N))


def build_neuro_ilu_csr(L_edge_index, L_values, U_edge_index, U_values, N, dirichlet_mask=None):
    """Build scipy CSR factors for a fixed Neuro-ILU preconditioner."""
    L_csr = edge_to_csr(L_edge_index, L_values, N)
    U_csr = edge_to_csr(U_edge_index, U_values, N)

    if dirichlet_mask is None or not dirichlet_mask.any().item():
        return L_csr, U_csr

    boundary_nodes = torch.where(dirichlet_mask)[0].detach().cpu().numpy()
    L_mod = L_csr.tolil()
    U_mod = U_csr.tolil()
    for dn in boundary_nodes:
        L_mod[dn, :] = 0
        L_mod[:, dn] = 0
        L_mod[dn, dn] = 1.0
        U_mod[dn, :] = 0
        U_mod[:, dn] = 0
        U_mod[dn, dn] = 1.0

    return L_mod.tocsr(), U_mod.tocsr()


def build_neuro_fsai_csr(G_L_edge_index, G_L_values, G_U_edge_index, G_U_values, N, dirichlet_mask=None):
    """Build scipy CSR factors for a fixed Neuro-FSAI preconditioner."""
    # FSAI factors are emitted in PyG source->target convention.  Matrix rows
    # are targets and columns are sources, matching the training _spmv path.
    G_L_csr = csr_matrix((
        G_L_values.detach().cpu().numpy().astype(np.float64),
        (G_L_edge_index[1].detach().cpu().numpy(),
         G_L_edge_index[0].detach().cpu().numpy())),
        shape=(N, N))
    G_U_csr = csr_matrix((
        G_U_values.detach().cpu().numpy().astype(np.float64),
        (G_U_edge_index[1].detach().cpu().numpy(),
         G_U_edge_index[0].detach().cpu().numpy())),
        shape=(N, N))

    if dirichlet_mask is None or not dirichlet_mask.any().item():
        return G_L_csr, G_U_csr

    boundary_nodes = torch.where(dirichlet_mask)[0].detach().cpu().numpy()
    G_L_mod = G_L_csr.tolil()
    G_U_mod = G_U_csr.tolil()
    for dn in boundary_nodes:
        G_L_mod[dn, :] = 0
        G_L_mod[:, dn] = 0
        G_L_mod[dn, dn] = 1.0
        G_U_mod[dn, :] = 0
        G_U_mod[:, dn] = 0
        G_U_mod[dn, dn] = 1.0

    return G_L_mod.tocsr(), G_U_mod.tocsr()


def apply_sparse_preconditioner(rhs, L_csr, U_csr):
    """Apply M^{-1} rhs = U^{-1} L^{-1} rhs with sparse triangular solves."""
    w = spsolve_triangular(L_csr, rhs, lower=True, unit_diagonal=True)
    return spsolve_triangular(U_csr, w, lower=False, unit_diagonal=False)


def apply_fsai_preconditioner(rhs, G_L_csr, G_U_csr):
    """Apply y = G_U @ (G_L @ rhs) with sparse matvecs."""
    return G_U_csr @ (G_L_csr @ rhs)


def bicgstab_sparse(A_csr, b_np, L_csr, U_csr, tol=1e-8, max_iter=2000):
    """BiCGSTAB with a fixed sparse Neuro-ILU preconditioner."""
    x = np.zeros(A_csr.shape[0], dtype=np.float64)
    r = b_np.ravel().astype(np.float64).copy() - A_csr @ x
    r_hat = r.copy()

    rho_prev = alpha = omega = 1.0
    v = np.zeros_like(r)
    p = np.zeros_like(r)
    b_norm = np.abs(b_np).max()

    for i in range(max_iter):
        rho = np.dot(r_hat, r)
        if abs(rho) < 1e-30:
            return max_iter, {}

        if i == 0:
            p[:] = r
        else:
            beta = (rho / rho_prev) * (alpha / omega)
            p = r + beta * (p - omega * v)

        p_hat = apply_sparse_preconditioner(p, L_csr, U_csr)
        v = A_csr @ p_hat
        r_hat_dot_v = np.dot(r_hat, v)
        if abs(r_hat_dot_v) < 1e-30:
            return max_iter, {}

        alpha = rho / r_hat_dot_v
        s = r - alpha * v
        if np.abs(s).max() <= b_norm * tol:
            x += alpha * p_hat
            return i + 1, {}

        s_hat = apply_sparse_preconditioner(s, L_csr, U_csr)
        t = A_csr @ s_hat
        t_dot_t = np.dot(t, t)
        omega = 0.0 if abs(t_dot_t) < 1e-30 else np.dot(t, s) / t_dot_t

        x += alpha * p_hat + omega * s_hat
        r = s - omega * t
        if np.abs(r).max() <= b_norm * tol:
            return i + 1, {}
        if abs(omega) < 1e-30:
            return max_iter, {}

        rho_prev = rho

    return max_iter, {}


def apply_neuro_ilu_preconditioner(r, L_edge_index, L_values, U_edge_index, U_values, N, dirichlet_mask=None):
    """Apply a fixed Neuro-ILU preconditioner to a residual vector."""
    L_csr, U_csr = build_neuro_ilu_csr(
        L_edge_index, L_values, U_edge_index, U_values, N, dirichlet_mask=dirichlet_mask)
    rhs = r.detach().cpu().numpy().reshape(-1)
    y = apply_sparse_preconditioner(rhs, L_csr, U_csr)
    return torch.from_numpy(y).to(device=r.device, dtype=r.dtype).reshape_as(r)


def apply_neuro_fsai_preconditioner(r, G_L_edge_index, G_L_values, G_U_edge_index, G_U_values, N, dirichlet_mask=None):
    """Apply a fixed Neuro-FSAI preconditioner to a residual vector."""
    G_L_csr, G_U_csr = build_neuro_fsai_csr(
        G_L_edge_index, G_L_values, G_U_edge_index, G_U_values, N, dirichlet_mask=dirichlet_mask)
    rhs = r.detach().cpu().numpy().reshape(-1)
    y = apply_fsai_preconditioner(rhs, G_L_csr, G_U_csr)
    return torch.from_numpy(np.asarray(y)).to(device=r.device, dtype=r.dtype).reshape_as(r)
