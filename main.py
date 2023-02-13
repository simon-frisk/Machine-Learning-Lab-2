import numpy as np
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import kernel
import plot
import test_data

N = 40
inputs, targets, classA, classB = test_data.generate(N)
plot.plot(classA, classB)
print(inputs)
