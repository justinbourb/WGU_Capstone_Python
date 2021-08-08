'''
Purpose: This file performs k-means clustering on a dataset
Input: a dataset in csv format
Output: Desired: 1) k-means clustering of data
                 2) a graph of clustered data
                 3) Assigning cluster # to the original data set for further machine learning uses
        Actual: ?
Results:
Source: https://realpython.com/k-means-clustering-python/#how-to-build-a-k-means-clustering-pipeline-in-python
'''

import tarfile
import urllib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

'''
Download and extract the TCGA dataset from UCI:
'''


def source_data_from_UCI():
    uci_tcga_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/"
    archive_name = "TCGA-PANCAN-HiSeq-801x20531.tar.gz"

    # Build the url
    full_download_url = urllib.parse.urljoin(uci_tcga_url, archive_name)

    # Download the file
    r = urllib.request.urlretrieve(full_download_url, archive_name)

    # Extract the data from the archive
    tar = tarfile.open(archive_name, "r:gz")
    tar.extractall()
    tar.close()


'''
The KMeans class in scikit-learn requires a NumPy array as an argument. The NumPy package has a 
helper function to load the data from the text file into memory as NumPy arrays:
'''


def load_and_format_data():
    datafile = "data/iqr_cleaned_data.csv"


    labels_file = "TCGA-PANCAN-HiSeq-801x20531/labels.csv"

    data = np.genfromtxt(
        datafile,
        delimiter=",",
        usecols=range(1, 20532),
        skip_header=1
    )

    true_label_names = np.genfromtxt(
        labels_file,
        delimiter=",",
        usecols=(1,),
        skip_header=1,
        dtype="str"
    )
    return data, true_label_names


def label_encoder_func(true_label_names):
    '''
    To use these labels in the evaluation methods, you first need to convert the abbreviations to integers
     with LabelEncoder:
    '''
    label_encoder = LabelEncoder()
    true_labels = label_encoder.fit_transform(true_label_names)
    true_labels[:5]

    '''
    Since the label_encoder has been fitted to the data, you can see the unique classes represented using 
    .classes_. Store the length of the array to the variable n_clusters for later use:
    '''
    label_encoder.classes_
    n_clusters = len(label_encoder.classes_)
    return label_encoder, n_clusters, true_labels


def create_pipline(n_clusters):
    '''
    Your gene expression data aren’t in the optimal format for the KMeans class, so you’ll need to build
    a preprocessing pipeline. The pipeline will implement an alternative to the StandardScaler class called
    MinMaxScaler for feature scaling. You use MinMaxScaler when you do not assume that the shape of all your
    features follows a normal distribution.

    The next step in your preprocessing pipeline will implement the PCA class to perform dimensionality reduction:
    '''
    preprocessor = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("pca", PCA(n_components=2, random_state=42)),
        ]
    )
    '''
    Now that you’ve built a pipeline to process the data, you’ll build a separate pipeline to perform k-means 
    clustering. You’ll override the following default arguments of the KMeans class:
    
    init: You’ll use "k-means++" instead of "random" to ensure centroids are initialized with some distance 
        between them. In most cases, this will be an improvement over "random".
    n_init: You’ll increase the number of initializations to ensure you find a stable solution.
    max_iter: You’ll increase the number of iterations per initialization to ensure that k-means will converge.
    
    Build the k-means clustering pipeline with user-defined arguments in the KMeans constructor:
    '''
    clusterer = Pipeline(
        [
            (
                "kmeans",
                KMeans(
                    n_clusters=n_clusters,
                    init="k-means++",
                    n_init=50,
                    max_iter=500,
                    random_state=42,
                ),
            ),
        ]
    )

    '''
    The Pipeline class can be chained to form a larger pipeline. Build an end-to-end k-means clustering
    pipeline by passing the "preprocessor" and "clusterer" pipelines to Pipeline:
    '''
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("clusterer", clusterer)
        ]
    )
    '''
    Calling .fit() with data as the argument performs all the pipeline steps on the data:
    '''
    pipe.fit(data)

    return pipe


def evaluate_pipe(pipe, true_labels):
    '''
    The pipeline performs all the necessary steps to execute k-means clustering on the gene expression data!
    Depending on your Python REPL, .fit() may print a summary of the pipeline. Objects defined inside pipelines
    are accessible using their step name.

    Evaluate the performance by calculating the silhouette coefficient:
    '''
    preprocessed_data = pipe["preprocessor"].transform(data)
    predicted_labels = pipe["clusterer"]["kmeans"].labels_
    print('pipe silhouette score:')
    print(silhouette_score(preprocessed_data, predicted_labels))

    '''
    The scale for each of these clustering performance metrics ranges from -1 to 1. A silhouette coefficient 
    of 0 indicates that clusters are significantly overlapping one another, and a silhouette coefficient of 
    1 indicates clusters are well-separated. An ARI score of 0 indicates that cluster labels are randomly 
    assigned, and an ARI score of 1 means that the true labels and predicted labels form identical clusters.

    Calculate ARI(Adjusted Rand Index), too, since the ground truth cluster labels are available:
    '''
    print('Adjusted Rand Index:')
    print(adjusted_rand_score(true_labels, predicted_labels))


def plot_clusters(data, pipe, label_encoder, true_labels):
    """
    The visual representation of the clusters confirms the results of the two clustering evaluation metrics.
    The performance of your pipeline was pretty good. The clusters only slightly overlapped, and cluster
    assignments were much better than random.

    Plot the results using a pandas DataFrame and the seaborn plotting library:
    """
    pcadf = pd.DataFrame(
        pipe["preprocessor"].transform(data),
        columns=["component_1", "component_2"],
    )

    pcadf["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_
    pcadf["true_label"] = label_encoder.inverse_transform(true_labels)

    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(8, 8))

    scat = sns.scatterplot(
        "component_1",
        "component_2",
        s=50,
        data=pcadf,
        hue="predicted_cluster",
        style="true_label",
        palette="Set2",
    )

    scat.set_title(
        "Clustering results from TCGA Pan-Cancer\nGene Expression Data"
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    plt.show()


if __name__ == "__main__":
    print('Start of main statement')
    data, true_label_names = load_and_format_data()
    '''
    Check out the first three columns of data for the first five samples as well as the labels 
    for the first five samples:
    '''
    print('First five rows of data')
    print(data[:5, :3])
    print('Ture label names')
    print(true_label_names[:5])

    label_encoder, n_clusters, true_labels = label_encoder_func(true_label_names)
    pipe = create_pipline(n_clusters)
    evaluate_pipe(pipe, true_labels)
    #plot_clusters(data, pipe, label_encoder, true_labels)
    print('End of main statement')
