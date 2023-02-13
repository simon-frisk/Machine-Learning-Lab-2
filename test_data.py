import numpy as np


def generate_test_data(N):
    inputs = np.random.randn(2, N)
    targets = np.random.randn(1, N)
    return inputs, targets
