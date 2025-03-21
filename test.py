from numpy import random
import numpy as np

random.seed(1235)
S1 = random.choice(4, size=3, replace=False)
S2 = random.choice(4, size=3, replace=False)



A = random.randint(0, 100, 27)
A = A.reshape((3,3,3))



X = random.randn(9)
X = X.reshape((3,3))

f1 = lambda A_i : np.trace(A_i @ X)
Y = np.array([f1(A[k]) for k in range(A.shape[0])])  

print(Y)

""" 
F1_F2 : results = {'1': 22, '5': 73, '3': 68, '7': 77, '4': 75, '6': 79, '2': 70, '8': 79, '9': 78, '11': 80, '10': 82, '13': 90, '12': 91}
F1 : results = {'1': 9, '3': 66, '4': 75, '6': 75, '7': 75, '5': 75, '2': 66, '8': 75, '9': 74, '11': 74, '10': 81, '13': 81, '12': 83} 
"""