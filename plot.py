import matplotlib.pyplot as plt
import numpy as np


def plot(a, indicator, classA, classB):
    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)

    grid = np.array([[indicator(a, [x, y]) for x in xgrid] for y in ygrid])

    plt.axis("Equal")
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
    plt.savefig("svmplot.pdf")
    plt.show()

