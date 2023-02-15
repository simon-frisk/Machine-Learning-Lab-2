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

# Get kernel
kernel = kernel_collection.polynomial(3)

# Get constraint
C = 100

# Get P matrix
def get_P(N, targets, inputs, kernel):
    P = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            P[i, j] = targets[i]*targets[j]*kernel(inputs[i, :], inputs[j, :])
    return P

# Define helper functions
def zerofun(a):
    return np.dot(a, targets)


def get_non_zeros(a):
    results = []
    for i, val in enumerate(a):
        if val > 1e-5:
            results.append({
                'a': val,
                'x': inputs[i, :],
                't': targets[i]
            })
    return results


def compute_bias(a):
    non_zeros = get_non_zeros(a)

    support_vector = non_zeros[0]['x']
    #support_vector_indicator = indicator(a, support_vector)

    sum = 0

    for i in range(N):
        sum += a[i]*targets[i]*kernel(support_vector, inputs[i, :])

    return sum - non_zeros[0]['t']


def indicator(a, s):
    sum = 0
    for i in range(N):
        sum += a[i]*targets[i]*kernel(s, inputs[i, :])
    return sum - compute_bias(a)


def master(classA, classB, inputs, targets, N, kernel, C):
    P = get_P(N, targets, inputs, kernel)
    bounds = [(0, C) for b in range(N)]

    def objective(a):
        return 0.5 * np.dot(np.transpose(a), np.dot(P, a)) - np.sum(a)
    
    results = minimize(objective, np.random.randn(N), bounds=bounds,
                   constraints={'type': 'eq', 'fun': zerofun})

    alphas = results['x']

    plot.plot(alphas, indicator, classA, classB)


# Optimize
#results = minimize(objective, np.random.randn(N), bounds=bounds,
#                   constraints={'type': 'eq', 'fun': zerofun})
#alphas = results['x']

# Extract support vectors


master(classA, classB, inputs, targets, N, kernel, C)
