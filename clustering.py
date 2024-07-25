import numpy as np
from sklearn import cluster 
from enum import Enum

import dataset 
import visualise as visu

class CLUSTERING_METHOD(Enum):
    KMEANS = 1
    AGGLOMERATIVE = 2
    DBSCAN = 3
    NEURALNETWORK = 4
    DYD2 = 5

def get_model(method):
    if method is CLUSTERING_METHOD.KMEANS:
        return cluster.KMeans(n_clusters=3, init='k-means++')
    elif method is CLUSTERING_METHOD.AGGLOMERATIVE:
        return cluster.AgglomerativeClustering(n_clusters=3, linkage="single")
    elif method is CLUSTERING_METHOD.DBSCAN:
        return cluster.DBSCAN(eps=0.05, min_samples=1)
    else:
        return None


def test_model(data, model):
    model.fit(data)
    labels = model.labels_
    visu.plot_clustering(data, labels)


if __name__ == "__main__":
    data_path = "./dataset/benchmark_artificial/smile1.arff"
    data = dataset.get_dataset(data_path)
    method = CLUSTERING_METHOD.DBSCAN

    test_model(data,get_model(method))

