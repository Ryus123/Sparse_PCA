# =============================================================================
# Imports
# ============================================================================= 
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import cvxpy as cp

# =============================================================================
# Setting
# ============================================================================= 
random.seed(12345)


# =============================================================================
# Functions
# ============================================================================= 
# Question 1)
def random_instance(k:int, m:int, n:int):
    """_summary_

    Args:
        k (int): cardinality of S1 and S2
        m (int): Number of observation
        n (int): dimension of the matrices

    Returns:
        A : 3D-np.aaray wich contains all A_i matrices of size nxn
        Y : 3D-np.aaray wich contains all y_i matrices given by <A_i, X>
    """
    n_2 = n**2
    #selects S1, S2 uniformly
    S1 = random.choice(n, size=k, replace=False)
    S1_C = np.setdiff1d(np.arange(n), S1, assume_unique=True) #Complement of S1
    S2 = random.choice(n, size=k, replace=False)
    S2_C = np.setdiff1d(np.arange(n), S2, assume_unique=True) #Complement of S2
    #Sample u,v
    u = np.multiply(0.01*np.random.randn(n), random.randint(1, n_2, n))
    u[S1_C] = np.zeros(n-k)
    v = np.multiply(0.01*np.random.randn(n), random.randint(1, n_2, n))
    v[S2_C] = np.zeros(n-k)
    # Compute X
    u = u.reshape(n,1)
    v = v.reshape(n,1)
    X = np.dot(u, v.T)
    # Compute A = {A1, .., Am}
    A = random.randn(m*n*n)
    A = A.reshape((m,n,n))
    # Compute Y = {y1, ..., ym}
    f1 = lambda A_i : np.trace(A_i @ X)
    Y = np.array([f1(A[k]) for k in range(m)])  
    
    return X, S1, S2, A, Y


# Question 3) c)
def solveur_nuc_l1(lbda:float, A:np.array, Y:np.array, beta=1):
    m, n, _ = A.shape
    Z = cp.Variable((n, n))
    F = beta*cp.norm(Z, "nuc") + lbda*cp.norm(Z, 1)
    constraints = [cp.trace(A[k] @ Z) == Y[k] for k in range(m)]
    objective = cp.Minimize(F)
    problem = cp.Problem(objective,constraints)
    # Solve it
    problem.solve(solver=cp.SCS)
    
    return Z.value


# Question 3) d)
def support_recovery_error(X, S1, S2, Z):
    # --- Find min_X
    min_X = np.min(np.abs(X[S1, :][:, S2]))
    # --- Support S1 x S2
    supp_S1_S2 = [(s1, s2) for s1 in S1.tolist() for s2 in S2.tolist()]
    supp_S1_S2 = set(supp_S1_S2)
    # --- Support of Z
    supp_Z = []
    # NOT in S1xS2
    indx_sup = np.argwhere(np.abs(Z) > (min_X/10)) 
    supp_Z.extend(idx for idx in indx_sup.tolist() if not tuple(idx) in supp_S1_S2)
    # in  S1xS2
    indx_inf = np.argwhere(np.abs(Z) < (min_X/10)) 
    supp_Z.extend(idx for idx in indx_inf.tolist() if tuple(idx) in supp_S1_S2)
    
    return len(supp_Z)
    

# Question 3) g)
def monte_carlo_support_error(k, nb_iter = 30, beta=1, lbda=0.1):
    error_M = 0
    for M in range(1000):
        X, S1, S2, A, Y = random_instance(k=k, m=M, n=20)
        for _ in range(nb_iter):
            Z = solveur_nuc_l1(lbda, A, Y, beta)
            error_M += (1/nb_iter) * (support_recovery_error(X, S1, S2, Z) == 0)
        
        if error_M >= 0.5:
            return k,M