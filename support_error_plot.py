#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
--- Here the description.

Created on ---HERE THE DATE Month/ DAY / YEARS.

Last update ---HERE THE DATE Month/ DAY / YEARS.

@author: E. DELAR
"""

# =============================================================================
# Imports
# ============================================================================= 
import numpy as np
import matplotlib.pyplot as plt
from utils import random_instance, solveur_nuc_l1, support_recovery_error

# =============================================================================
# Script
# ============================================================================= 
# Question 3) f)
X, S1, S2, A, Y = random_instance(k=3, m=20, n=6)
errors = {'support recovery error' : [], 'Frobenius error' : []}
domaine = np.logspace(-5, 1, num=100)

for lbda in domaine:
    Z = solveur_nuc_l1(lbda, A, Y)
    errors['support recovery error'].append(support_recovery_error(X, S1, S2, Z))
    errors['Frobenius error'].append(np.linalg.norm(X - Z, "fro"))
    
plt.set_cmap("RdPu")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(domaine, errors["support recovery error"], lw=2)
axes[0].set_title("support recovery error", fontsize=16)
axes[0].set_xscale("log")
axes[0].set_xlabel("Lambda", fontsize=14)
axes[0].set_ylabel("error", fontsize=14)
axes[0].legend(loc=3)

axes[1].plot(domaine, errors["Frobenius error"] , lw=2)
axes[1].set_title("Frobenius error", fontsize=16)
axes[1].set_xscale("log")
axes[1].set_xlabel("Lambda", fontsize=14)
axes[1].set_ylabel("Error", fontsize=14)
axes[1].legend(loc=3)

plt.savefig("plot/support_error_plot.png")