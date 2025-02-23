#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
--- Here the description.

Created on ---HERE THE DATE Month/ DAY / YEARS.

Last update ---HERE THE DATE Month/ DAY / YEARS.

@author: E. DELAR
"""

### --- Import
import numpy as np
from numpy import random
import math
import scipy
from scipy import linalg
import cvxpy as cp

### --- Setting
random.seed(12345)


### --- Function

# Question 1)
def random_instance(k, m, n):
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
    f1 = lambda A_i,_ : np.dot(A_i, X)
    Y = np.apply_over_axes(f1, A, 0)
    return X, S1, S2, A, Y



### --- Test
# print(random_instance(3, 2, 5))