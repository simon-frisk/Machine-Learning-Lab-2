import numpy as np
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import kernel as kernel_collection
import plot
import test_data

np.random.seed(100)

# Get data
N = 40
inputs, targets, classA, classB = test_data.generate(N)
#inputs = np.array([[1, 0], [-1, 0]])
#targets = np.array([1, -1])

# Get kernel
kernel = kernel_collection.radial_basis_function(0.2)

# Get constraint
C = 100
bounds = [(0, C)  for b in range(N)]

# Get P matrix
P = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        P[i, j] = targets[i]*targets[j]*kernel(inputs[i, :], inputs[j, :])

def objective(a):
    return 0.5 * np.dot(np.transpose(a), np.dot(P, a)) - np.sum(a)

def zerofun(a):
    return np.dot(a, targets)

def indicator(a, s):
    sum = 0
    for i in range(N):
        sum += a[i]*targets[i]*kernel(s, inputs[i, :]) 
    return sum # - b


results = minimize(objective, np.random.randn(N), bounds=bounds, constraints={'type':'eq', 'fun':zerofun})
alphas = results['x']


plot.plot(alphas, indicator, classA, classB)



