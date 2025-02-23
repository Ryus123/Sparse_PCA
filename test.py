from numpy import random
import numpy as np

random.seed(1235)
S1 = random.randint(-10, 11, 3)
S2 = random.randint(1, 11, 3)



A = random.randn(5*2*2)
A = A.reshape((5,2,2))

for i in range(5):
    A[i] = np.ones((2,2))*i


test = 1+np.random.randn(2, 2)

f1 = lambda x,_ : np.dot(x, test)

Y = np.apply_over_axes(f1, A, 0)

print(Y.shape)

for i in range(5):
    print(np.dot(Y[i], np.linalg.inv(test)))