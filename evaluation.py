# Silhouette score, indice de Davies Bouldin, indice de Calinski-Harabasz (variance ratio criterion), indice de Dunn

import numpy as np
import sklearn.metrics as metrics
from enum import Enum
import math


def euclidean(sample1, sample2):
    """Caculate the euclidean distance between two data point.
    The formula is:
        dist(X,Y) = \sqrt{\sum_{i=1}^{n}{ ( x_i - y_i ) ^ 2}}

    Args:
        sample1 (list of float): Features assigned to the first sample
        sample2 (list of float): Features assigned to the second sample

    Returns:
        distance (float): Euclidean distance between the two samples
    """
    sum = 0
    for idx in range(len(sample1)):
        sum += (sample1[idx] - sample2[idx])**2
    return math.sqrt(sum)


def find_nearest_cluster(dataset, labels, sample_label, sample):
    """Find the nearest cluster available from a given sample, outside its own cluster.
    This function uses a distance function to evaluate the distance from 
    a given sample and all other clusters in the dataset.
    The label of the cluster minimising the distance with the given sample is then returned
    
    Args:
        dataset (list of [float]): List containing all the samples. A sample is a list of features 
        labels (list of int): Label associated with each sample in the dataset
        sample_label (int): label of the sample for which we have to find the closest cluster
        sample (list of float): sample features

    Returns:
        label (int): ID of the closest cluster
    """
    unique_labels, labelled_samples = get_labelled_data(dataset, labels)

    distances = []
    for idx in range(len(unique_labels)):
        if unique_labels[idx] == sample_label:
            distances.append(-1)
        else:
            dist = 0
            for s in labelled_samples[idx]:
                dist += euclidean(s, sample)
            distances.append(dist/len(labelled_samples[idx]))
    
    min_dist = np.min([dist for dist in distances if dist > 0])
    closest_cluster = unique_labels[distances.index(min_dist)]

    return closest_cluster

def get_labelled_data(dataset, labels):
    """Arrange a dataset depending on the available clusters.
    It groups all sample issued from a similar cluster together.
    By doing so, it is possible to work on each cluster contained in a dataset.

    Args:
        dataset (list of [float]): List containing all the samples. A sample is a list of features 
        labels (list of int): Label associated with each sample in the dataset

    Returns:
        unique_labels (list of int): Ordered list of labels available in the dataset
        labelled_sample (list of list of [float]): List containing the dataset clusters discretised. 
                                                   Each list corresponds to a unique cluster.
                                                   The cluster list contains all sample related to this cluster.
    """
    unique_labels = np.unique(labels)
    labelled_samples = []

    for label in unique_labels:
        labelled_samples.append([s for s,l in zip(dataset,labels) if l==label])
    return unique_labels, labelled_samples

def get_centroid(cluster):
    """Get the centroid point of a cluster.
    The centroid is taken by summing each feature of all samples in the cluster, and then taking the average.

    Args:
        cluster (list of [float]): Samples contained in the cluster.

    Returns:
        centroid [list of float]: Centroid coordinates for each dimension.
    """
    sample_count = len(cluster)
    centroid = [sum(f)/sample_count for f in zip(*cluster)]
    return centroid

def silhouette_score(dataset, labels):
    """Compute the silhouette score for a clustered dataset.
    The silhouette score for a unique sample measures how well this sample is assigned to its cluster by comparing it with other clusters.
    Its range from [-1, 1], with a higher score indicating that a sample is well assigned to its cluster.
    The silhouette score of an entire dataset is the average of the silhouette score for all samples. 

    It is defined as follow:
        First, we evaluate how well the sample i is assigned to its cluster C_I:
            score_{intra}(i) = \frac{1}{|C_I| - 1} \sum_{j \in C_I, i \neq j} dist(i,j)
        
        Then, we evaluate the dissimilarities between the sample i and an other cluster C_J.
        The goal is to find the least dissimilar cluster from the sample i (aka, the "neighboring cluster"), hence:
            score_{inter}(i) = \min_{J \neq I} \frac{1}{|C_J|} \sum_{j \in C_J} dist(i,j)
        
        Finally, the silhouette score for a unique sample is defined as:
            silhouette(i) = \frac{score_{inter}(i) - score_{intra}(i)}{\max\{score_{intra}(i), score_{inter}(i)\}}
            
        The silhouette score of a whole clustere dataset can be defined as:
            silhouette = \frac{1}{N} \sum_{i=1}^{N} silhouette(i)

    Args:
        dataset (list of [float]): List containing all the samples. A sample is a list of features 
        labels (list of int): Label associated with each sample in the dataset

    Returns:
        silhouette score (int): Silhouette score for the entire clustered dataset.
    """
    sample_score = []
    for i in range(len(dataset)):
        sample_label = labels[i]
        current_sample = dataset[i]
        nearestcluster = find_nearest_cluster(dataset, labels, sample_label, current_sample)
        
        intracluster_score, nearestcluster_score = 0, 0
        
        # intra_score = (1 / |Ci|-1) * sum( dist(i,j) ) with Ci cluster of i
        idx = 0
        for sample in [s for s,l in zip(dataset,labels) if l==sample_label]:
            if(sample is not current_sample):
                intracluster_score += euclidean(current_sample, sample)
            idx += 1
        intracluster_score /= (idx-1)
        
        # nearest_score = (1 / |Cj|) * sum( dist(i,j) ) with Cj nearest cluster from i
        idx = 0
        for sample in [s for s,l in zip(dataset,labels) if l==nearestcluster]:
            nearestcluster_score += euclidean(current_sample, sample)
            idx += 1
        nearestcluster_score /= idx
        
        score = (nearestcluster_score - intracluster_score)/(max(intracluster_score, nearestcluster_score))
        sample_score.append(score)

    return np.mean(sample_score)

def BCSS(dataset, labels):
    """_summary_

    Args:
        dataset (list of [float]): List containing all the samples. A sample is a list of features 
        labels (list of int): Label associated with each sample in the dataset

    Returns:
        _type_: _description_
    """
    _, labelled_samples = get_labelled_data(dataset, labels)
    data_centroid = get_centroid(dataset)

    cluster_score = []
    for cluster in labelled_samples:
        sample_count = len(cluster)
        cluster_centroid = get_centroid(cluster)
        cluster_score.append(sample_count * (euclidean(data_centroid, cluster_centroid)**2))
    return np.sum(cluster_score)

def WCSS(dataset, labels):
    """_summary_

    Args:
        dataset (list of [float]): List containing all the samples. A sample is a list of features 
        labels (list of int): Label associated with each sample in the dataset

    Returns:
        _type_: _description_
    """
    _, labelled_samples = get_labelled_data(dataset, labels)

    cluster_score = []    
    for cluster in labelled_samples:
        score = 0
        cluster_centroid = get_centroid(cluster)
        for sample in cluster:
            score += euclidean(cluster_centroid, sample)**2
        cluster_score.append(score)
    
    return np.sum(cluster_score)

def calinski_harabasz_score(dataset, labels):
    """_summary_

    Args:
        dataset (list of [float]): List containing all the samples. A sample is a list of features 
        labels (list of int): Label associated with each sample in the dataset

    Returns:
        _type_: _description_
    """
    sample_count = len(dataset)
    cluster_count = len(np.unique(labels))
    BCSS_score = BCSS(dataset,labels)
    WCSS_score = WCSS(dataset,labels)

    a = BCSS_score/(cluster_count - 1)
    b = WCSS_score/(sample_count - cluster_count)

    return a / b

def get_cluster_diameter(cluster):
    """_summary_

    Args:
        cluster (_type_): _description_

    Returns:
        _type_: _description_
    """
    diameter = 0
    centroid = get_centroid(cluster)
    for sample in cluster:
        diameter += euclidean(sample, centroid)
    diameter /= len(cluster)  
    
    return diameter

def cluster_similarity(dataset, labels):
    """_summary_

    Args:
        dataset (list of [float]): List containing all the samples. A sample is a list of features 
        labels (list of int): Label associated with each sample in the dataset

    Returns:
        _type_: _description_
    """
    _, labelled_samples = get_labelled_data(dataset, labels)

    similarities = []
    for idx in range(len(labelled_samples)):
        current_cluster = labelled_samples[idx]
        current_centroid = get_centroid(current_cluster)

        # Calculate cluster diameter -> The average distance between each samples in the cluster with its centroid
        intra_cluster_diameter = get_cluster_diameter(current_cluster)
        
        other_clusters = [clust for i,clust in enumerate(labelled_samples) if i!=idx]
        cluster_similarity = []
        for clust in other_clusters:
            # Calculate distance between the the two clusters
            other_cluster_centroid = get_centroid(clust) 
            centroids_distance = euclidean(current_centroid, other_cluster_centroid) 
            
            # Calculate the second cluster diameter
            other_cluster_diameter = get_cluster_diameter(clust)

            # Calculate the similarity score between two clusters
            score = (intra_cluster_diameter + other_cluster_diameter) / centroids_distance
            cluster_similarity.append(score)
        similarities.append(cluster_similarity)

    return similarities

def davies_bouldin_index(dataset, labels):
    """_summary_

    Args:
        dataset (list of [float]): List containing all the samples. A sample is a list of features 
        labels (list of int): Label associated with each sample in the dataset

    Returns:
        _type_: _description_
    """
    similarities = cluster_similarity(dataset, labels)
    cluster_count = len(np.unique(labels))
    max_similarities = [max(sim) for sim in similarities]

    return (np.sum(max_similarities)) / cluster_count
    
if __name__ == "__main__":
    import clustering as clust
    import dataset
    data_path = "./dataset/benchmark_artificial/smile1.arff"
    data = dataset.get_dataset(data_path)
    method = clust.CLUSTERING_METHOD.KMEANS
    model = clust. get_model(method)
    model.fit(data)
    labels = model.labels_

    score = silhouette_score(data, labels)
    score_sklearn = metrics.silhouette_score(data, labels)
    print(f"Silhouette score: {round(score,6)} / {round(score_sklearn,6)}")

    score = calinski_harabasz_score(data, labels)
    score_sklearn = metrics.calinski_harabasz_score(data, labels)
    print(f"calinski_harabasz_score: {round(score,6)} / {round(score_sklearn,6)}")

    score = davies_bouldin_index(data, labels)
    score_sklearn = metrics.davies_bouldin_score(data, labels)
    print(f"davies_bouldin_index: {round(score,6)} / {round(score_sklearn,6)}")