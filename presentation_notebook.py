import numpy as np
import random


def generate(N):
    classA = np.concatenate((np.random.randn(
        int(N/4), 2) * 0.2 + [1.5, 0.5], np.random.randn(int(N/4), 2) * 0.2 + [-1.5, 0.5]))
    classB = np.random.randn(int(N/2), 2) * 0.2 + [0, -0.5]

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((
        np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    return inputs, targets, classA, classB
