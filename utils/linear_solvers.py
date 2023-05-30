"""
Module for wrapping different linear solvers.
"""
import numpy as np
import scipy.sparse as sps
try:
    from petsc4py import PETSc
except ImportError:
    print("No PETSc. Bicstab solver wont work")

try:
    import pyamg
except ImportError:
    print("No pyamg. amg solver wont work")

try:
    import pypardiso
except ImportError:
    print("No pypardis. pardiso solver won't work")


def amg(A, b, tol=1e-10):
    """Solve with AMG."""
    B = None  # no near-null spaces guesses for SA

    # use AMG based on Smoothed Aggregation (SA) and display info
    mls = pyamg.smoothed_aggregation_solver(A, B=B)
    print(mls)
    # # Solve Ax=b with no acceleration ('standalone' solver)
    # standalone_residuals = []
    # x = mls.solve(b, tol=1e-10, accel=None, residuals=standalone_residuals)

    # Solve Ax=b with Conjugate Gradient (AMG as a preconditioner to CG)
    residuals = []
    x = mls.solve(b, tol=tol, accel="cg", residuals=residuals)
    if residuals[-1] > 100 * tol:
        Warning("Iterative solver failed. Falling back to direct solver")
        return umfpack(A, b)
    print("Solved linear system with AMG. Residual is: {}".format(residuals[-1]))
    return x

def pardiso(A, b):
    "Solve with pardiso"
    return pypardiso.spsolve(A, b)

def gmres(A, b, x0, tol=1e-10):
    "Solve with gmres"
    M_iLU = sps.linalg.spilu(A, fill_factor=20, drop_tol=1e-5)
    M = sps.linalg.LinearOperator(A.shape, M_iLU.solve)

    def callback(res):
        print("Gmres residual: {}".format(res))

    x, info = sps.linalg.gmres(
        A, b, x0=x0, M=M, tol=tol, maxiter=200, callback=callback
    )
    if info != 0:
        Warning("Iterative solver failed. Falling back to direct solver")
        return umfpack(A, b)
    return x

def bicstab(A, b, tol=1e-10, gb=None):
    "Solve with bicstab"
    if gb is None:
        P = sps.identity(A.shape[0])
    else:
        nc = gb.num_cells()
        p_idx = np.arange(0, nc, 1)
        c_idx = np.arange(nc, 2 * nc, 1)
        col = np.ravel(np.vstack((p_idx, c_idx)), order='F')
        row = np.arange(0, 2 * nc)
        data = np.ones(2 * nc, dtype=int)
        P = sps.coo_matrix((data, (row, col)))
        
#    Ao = A.copy()
    A = P * A * P.T
    if A.format != "csr":
        A = A.tocsr()

    M = PETSc.Mat().createAIJ(size=A.shape,
                          csr=(A.indptr, A.indices,
                               A.data))
    x, rhs = M.getVecs()
    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)
    ksp.setType('bcgs')
    ksp.getPC().setType('ilu')
    rhs.setArray(P * b)
    ksp.setOperators(M)
    ksp.setTolerances(1e-4, tol)
    ksp.solve(rhs, x)

#    print("linear it: ", ksp.getIterationNumber())
#    print("converged reason: ", ksp.getConvergedReason())
    return P.T * x.getArray()


def umfpack(A, b):
    "Solve with umfpack"
    if A.nnz > 500000:
        A.indices = A.indices.astype(np.int64)
        A.indptr = A.indptr.astype(np.int64)
    return sps.linalg.spsolve(A, b, use_umfpack=True)

def superlu(A, b):
    "Solve with superlu"
    return sps.linalg.spsolve(A, b)
