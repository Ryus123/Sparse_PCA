# =============================================================================
# Imports
# ============================================================================= 
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from multiprocessing import  cpu_count
from utils import monte_carlo_support_error

# =============================================================================
# Setting
# ============================================================================= 
np.random.seed(12345)

# =============================================================================
# Plot
# ============================================================================= 
results = {}
n_jobs = cpu_count() - 1

def f(x):
    return monte_carlo_support_error(x, beta=0, lbda=0.01)

if __name__ == '__main__': 
    with Pool(processes=n_jobs) as pool:
        for k, M in pool.imap_unordered(f, range(1, 14)):
            results[str(k)] = M
    
liste_k = [k for k in range(1,14)]
liste_M = [results[str(k)] for k in liste_k]

plt.figure(figsize=(8, 5))
plt.plot(liste_k, liste_M, label='F = F2')

# Labels et titre
plt.xlabel("k")
plt.ylabel("M")
plt.title("smallest value of m s.t., Problem reaches zero support recovery error")
plt.legend()
plt.grid()

plt.savefig("plot/plot_smallest_m_F2.png")