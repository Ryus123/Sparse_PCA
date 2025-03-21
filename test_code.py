# =============================================================================
# Imports
# ============================================================================= 
import numpy as np
import matplotlib.pyplot as plt
from utils import random_instance, solveur_nuc_l1, support_recovery_error

# =============================================================================
# Script
# ============================================================================= 
# Question 3) e)
lbda_values = np.linspace(0,1,10)  # Différentes valeurs de lambda
M_range = range(1, 21)  # Valeurs de M de 1 à 20

for lbda in lbda_values:
    errors = []
    for M in M_range:
        X, S1, S2, A, Y = random_instance(k=2, m=M, n=4)
        Z = solveur_nuc_l1(lbda, A, Y)
        error = support_recovery_error(X, S1, S2, Z)
        errors.append(error)
    
    plt.plot(M_range, errors, marker='o', label=f'lambda = {lbda:.4f}')

plt.xlabel("M")
plt.ylabel("Support recovery error")
plt.title("Support recovery error as a function of M for different lambda")
plt.legend()
plt.grid(True)
plt.savefig("plot/test_code.png")