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

def plot_dataset(data, title="Clustering dataset"):
    """Plot a 2D dataset

    Args:
        data (list of [float, float]): Dataset to ploat
        title (str, optional): Figure title. Defaults to "Clustering dataset".
    """
    f0 = [x[0] for x in data]
    f1 = [x[1] for x in data]
    plt.scatter(f0, f1)
    plt.grid(True)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def display_all_data(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path,f))]

    x_fig, y_fig = get_closest_multiply(len(files))
    fig, axes = plt.subplots(x_fig, y_fig, figsize=(30,30))
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
    plt.savefig(f"{folder_path}dataset_plot.png")

def get_closest_multiply(integer):
    a = int(sqrt(integer)) + 1
    while (integer % a != 0):
        a -= 1
    b = integer//a
    return a, b


if __name__ == "__main__":
    data = get_dataset("./dataset/benchmark_artificial/2d-20c-no0.arff")
    #plot_dataset(data)
    display_all_data("./dataset/benchmark_artificial/")