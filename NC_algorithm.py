# =============================================================================
# Imports
# ============================================================================= 
import numpy as np
import matplotlib.pyplot as plt
from utils import random_instance, solveur_nuc_l1

# =============================================================================
# Script
# ============================================================================= 
# Question 4) d)
def support_appoximation(A, Y, k, nb_iter=30):
    m,n,_ = A.shape
    Z = np.zeros((n,n))
    # Estime X
    for _ in range(nb_iter):
        Z += (1/nb_iter)*solveur_nuc_l1(.1, A, Y)
    # S1 : K rows with largest l1-norm
    rows_norm = np.sum(np.abs(Z), axis=1)
    S1 = np.argsort((-1)*rows_norm)[:k]
    # S2 : K rows with largest l1-norm
    cols_norm = np.sum(np.abs(Z), axis=0)
    S2 = np.argsort((-1)*cols_norm)[:k]
    
    return S1, S2

def f_loss(u, v, A, y):
    m, n, _ = A.shape
    loss = 0
    
    for i in range(m):
        loss += (1/(2*m))*(y[i] - v.T @ A[i] @ u )**2
    
    return loss

def grad_u(u, v, A, y):
    m, n, _ = A.shape
    grad = np.zeros(n)
    
    for i in range(m):
        error = y[i] - v.T @ A[i] @ u 
        grad -= (1/m)*error*(A[i] @ v) 
    
    return grad

def grad_v(u, v, A, y):
    m, n, _ = A.shape
    grad = np.zeros(n)
    
    for i in range(m):
        error = y[i] - v.T @ A[i] @ u 
        grad -= (1/m)*error*(A[i] @ u) 
    
    return grad

def support_constraint(x, S, n):
    for i in range(n):
        if i not in S:
            x[i] = 0
    return x

def non_convex_algo(S1, S2, nb_iter, lr, threshold, A, y):
    m, n, _ = A.shape
    u = np.zeros(n)
    v = np.zeros(n)
    
    for _ in range(nb_iter):
        u -= lr*grad_u(u,v, A, y)
        v -= lr*grad_v(u,v, A, y)
        u = support_constraint(u, S1, n)
        v = support_constraint(v, S2, n)
        if f_loss(u,v, A, y) <= threshold:
            break
    
    return u @ v.T
        
plt.figure()        
errors = []
domaine = [i for i in range(1,1000, 25)]
for M in domaine:
    X, S1, S2, A, Y = random_instance(k=4, m=M, n=10)
    S1_approx, S2_approx = support_appoximation(A, Y, k=4, nb_iter=30)
    Z = non_convex_algo(S1_approx, S2_approx, nb_iter=50, lr=.01, threshold=1e-6, A=A, y=Y)

    errors.append(np.linalg.norm(X - Z, 'fro'))
    
# Tracé de l'erreur en fonction de m
plt.plot(domaine, errors,)
plt.xlabel("M")
plt.ylabel("Frobenius norme of X - Z")
plt.title("Reconstruction errors as a function of M")
plt.grid()
plt.savefig("plot/NC_algo_differents_M.png")


plt.figure() 
errors = []
domaine = np.linspace(0,1,100)
for l in domaine:
    X, S1, S2, A, Y = random_instance(k=4, m=100, n=10)
    S1_approx, S2_approx = support_appoximation(A, Y, k=4, nb_iter=30)
    Z = non_convex_algo(S1_approx, S2_approx, nb_iter=50, lr=l, threshold=1e-6, A=A, y=Y)
    errors.append(np.linalg.norm(X - Z, 'fro'))
    
# Tracé de l'erreur en fonction de m
plt.plot(domaine, errors)
plt.xlabel("learning rate")
plt.ylabel("Frobenius norme of X - Z")
plt.title("Reconstruction errors as a function of learning rate")
plt.grid()
plt.savefig("plot/NC_algo_differents_lr.png")


plt.figure() 
errors = []
domaine = np.linspace(0,1,100)
for l in domaine:
    X, S1, S2, A, Y = random_instance(k=4, m=100, n=10)
    S1_approx, S2_approx = support_appoximation(A, Y, k=4, nb_iter=30)
    Z = non_convex_algo(S1_approx, S2_approx, nb_iter=50, lr=l, threshold=1e-6, A=A, y=Y)
    errors.append(np.linalg.norm(X - Z, 'fro'))
    
# Tracé de l'erreur en fonction de m
plt.plot(domaine, errors)
plt.xlabel("learning rate")
plt.ylabel("Frobenius norme of X - Z")
plt.title("Reconstruction errors as a function of learning rate for m=10")
plt.grid()
plt.savefig("plot/NC_algo_differents_lr_small_m.png")