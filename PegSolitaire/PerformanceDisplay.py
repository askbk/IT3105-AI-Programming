import matplotlib.pyplot as plt
import numpy as np


def graph_performance(scores):
    plt.plot(np.arange(len(scores)), scores)
    plt.draw()
    plt.show()
