import matplotlib.pyplot as plt


def plot(classA, classB):
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')

    plt.axis("Equal")
    plt.savefig("svmplot.pdf")
    plt.show()
