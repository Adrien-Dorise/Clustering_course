import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import os
from math import sqrt

def get_dataset(path):
    """Import an .arff data file into a Python variable.
    Note that if the dataset contains more than 2 features, only the first two are exported.

    Args:
        path (string): Path to the .arff file

    Returns:
        data (list of [float, float]): Imported dataset
    """
    data = arff.loadarff(open(path,'r'))
    data = [[float(x[0]),float(x[1])] for x in data[0]]
    return data

def plot_2Ddataset(data, title="Clustering dataset"):
    """Plot a 2D dataset

    Args:
        data (list of [float, float]): Dataset to ploat
        title (str, optional): Figure title. Defaults to "Clustering dataset".
    """
    if(len(data[0] < 2)):
        raise Warning(f"WARNING in plot_2Ddataset: Not enough features. feature count=({len(data[0])}) / required=(2)")
    
    f0 = [x[0] for x in data]
    f1 = [x[1] for x in data]
    plt.scatter(f0, f1)
    plt.grid(True)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def plot_3Ddataset(data, title="Clustering dataset"):
    """Plot a 3D dataset

    Args:
        data (list of [float, float]): Dataset to ploat
        title (str, optional): Figure title. Defaults to "Clustering dataset".
    """
    if(len(data[0] < 3)):
        raise Warning(f"WARNING in plot_3Ddataset: Not enough features. feature count=({len(data[0])}) / required=(3)")
    
    f0 = [x[0] for x in data]
    f1 = [x[1] for x in data]
    f2 = [x[2] for x in data]
    plt.scatter3D(f0, f1, f2)
    plt.grid(True)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def plot_benchmark_recap(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path,f))]

    x_fig, y_fig = get_closest_multiply(len(files))
    fig, axes = plt.subplots(x_fig, y_fig, figsize=(y_fig*3,x_fig*3))
    idx = 0
    for f in files:
        print(f)
        data = get_dataset(os.path.join(folder_path,f))
        f0 = [x[0] for x in data]
        f1 = [x[1] for x in data]
        axes.flat[idx].scatter(f0, f1)
        axes.flat[idx].label_outer()
        axes.flat[idx].set_title(f)
        idx += 1
    plt.savefig(f"{folder_path}plot/dataset_plot.png")

def get_closest_multiply(integer):
    a = int(sqrt(integer)) + 1
    while (integer % a != 0):
        a -= 1
    b = integer//a
    return a, b


if __name__ == "__main__":
    data = get_dataset("./dataset/benchmark_artificial/2d-20c-no0.arff")
    plot_2Ddataset(data)
    plot_benchmark_recap("./dataset/benchmark_artificial/")
