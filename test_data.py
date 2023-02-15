import numpy as np
import random


def generate(N, a1=[1.5, 0.5], a2=[-1.5, 0.5], b=[0, -0.5], deviation=0.2):
    classA = np.concatenate((np.random.randn(
        int(N/4), 2) * deviation + a1, np.random.randn(int(N/4), 2) * deviation + a2))
    classB = np.random.randn(int(N/2), 2) * deviation + b

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((
        np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    return inputs, targets, classA, classB
