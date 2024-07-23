import matplotlib.pyplot as plt
import numpy as np


def plot_clustering(dataset, labels, title="Clustering result"):
    f0 = [x[0] for x in dataset]
    f1 = [x[1] for x in dataset]
    fig = plt.scatter(f0, f1, c=labels, cmap="tab20")
    lab = [str(x) for x in np.unique(labels)]
    plt.legend(handles=fig.legend_elements()[0], labels=lab, title="clusters")
    plt.grid(True)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
