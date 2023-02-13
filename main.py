import numpy as np
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import kernel as kernel_collection
import plot
import test_data

N = 40
inputs, targets, classA, classB = test_data.generate(N)
p = 2
kernel = kernel_collection.polynomial(2)
C = 1
bounds = [(0, C)  for b in range(N)]

# Get P matrix
P = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        P[i, j] = targets[i]*targets[j]*kernel(inputs[i, :], inputs[j, :])

def objective(a):
    return 0.5 * np.dot(np.transpose(a), np.dot(P, a)) + np.sum(a)

def zerofun(a):
    return np.dot(a, targets)


results = minimize(objective, np.zeros(N), bounds=bounds, constraints={'type':'eq', 'fun':zerofun})
alphas = results['x']

print(alphas)
#plot.plot(classA, classB)




