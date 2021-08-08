"""
Purpose: This file will create data clusters and export the updated csv (including clusters).
Input: a csv file to be inspected
Return: none
Output: A csv file containing the original information plus a cluster column.
Lessons learned:
    1) KMeans analysis with Python
    2) Working with numpy, pandas and sklearn and how they interact
    3) The importance of standardizing data before clustering
        a) Large values skew clustering, all values should be similar (standardized) for best results
"""

from sklearn.cluster import KMeans
from sklearn import preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

'''
Printing the clusters:
source: https://stackoverflow.com/questions/28344660/how-to-identify-cluster-labels-in-kmeans-scikit-learn
sklearn scatter plots using kmeans analysis:
source: https://blog.floydhub.com/introduction-to-k-means-clustering-in-python-with-scikit-learn/
'''


def convert_csv_to_numpy_array(file):
    """
    Purpose: This function:
                1) loads a csv using pandas
                2) converts it to a pandas data frame
                3) converts it to a numpy array and returns the array
            A numpy array is required for sklearn.cluster Kmeans.fit()
    Reasoning: sklearn.cluster Kmeans relies on a numpy array ->
                a numpy array relies on a panda data frame ->
                a pandas data frame relies on pd.read_file (when using a csv)
                These packages all rely on one another, that is the base reasoning behind usage choice.

    :param file:
    :return a numpy array:
    """
    # loads the file, loads columns 2,3 only (price and item id), header row = 0 is not loaded
    data_set = pd.read_csv(file, usecols=[2, 3], header=0)
    data_frames = pd.DataFrame(data_set)

    required_array = np.array(data_frames.values)
    return required_array


def plot_clusters_no_legend(x, kmeans):
    """
    Purpose: This function creates a scatter plot from a numpy array.
        It colors each cluster a different color, but does not generate a legend.
    :param x: a numpy array
    :return: nothing, plots a scatter plot
    """
    # Plotting the cluster centers and the data points on a 2D plane
    # The data positions, x[:,0] = the first column, x[:,-1] = the second column
    # In this case x = price, y = # of items, c=predict colors the dots based on cluster number
    predict = kmeans.predict(x)
    plt.scatter(x[:, 0], x[:, -1], c=predict, label=["0", "1", "2"])
    plt.legend(loc='upper right')
    # This line adds the clusters into the scatter
    # The clusters are marked by a red x
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='x')

    # create labels for the scatter plot
    plt.title('Data points and cluster centroids')
    plt.xlabel('Item price (price)')
    plt.ylabel('# of items ordered (item_id)')
    # add a legend to the plot

    # show the scatter plot
    plt.show()


def plot_clusters_with_legend(x, kmeans):
    """
    Purpose: This function creates a scatter plot from a numpy array.
            It colors each cluster a different color and generates a legend.
    :param x: a numpy array
    :return: nothing, plots a scatter plot
    """
    # lists the number of clusters to be graphed, used later to color scatter plot
    categories = np.array(kmeans.labels_)

    # use red, green and blue color map, used later for scatter plot
    colormap = np.array(['#f00', '#0f0', '#00f'])
    # Plotting the the data points on a 2D plane
    # The data positions, x[:,0] = the first column, x[:,-1] = the second column
    # In this case x = price, y = # of items, c= colors the dots based on cluster number
    plt.scatter(x[:, 0], x[:, -1], c=colormap[categories])

    # pop_a,b,c matches colors to handles, which is later used to generate the legend for the scatter plot
    pop_a = mpatches.Patch(color='#f00', label='Population A')
    pop_b = mpatches.Patch(color='#0f0', label='Population B')
    pop_c = mpatches.Patch(color='#00f', label='Population C')

    # create labels for the scatter plot
    plt.title('Data points and cluster centroids')
    plt.xlabel('Item price (price)')
    plt.ylabel('# of items ordered (item_id)')
    # add a legend to the plot
    plt.legend(handles=[pop_a, pop_b, pop_c])
    # show the scatter plot
    plt.show()


def standardized_data(x):
    """
    Purpose: This function converts a numpy array into a standarized numpy array.
        Large data values skew the results of clustering, so standardization creates "better" clustering
    :param x: a numpy array
    :return: a standardized numpy array
    """
    standardized = preprocessing.scale(x)
    return standardized


def save_output_to_csv(kmeans):
    """
    Purpose: This functions adds the cluster assigment back into the original data
            and saves the combined information as a new csv file.
            This can later be used in further analysis.
    :param kmeans: kmeans analysis from sklearn.cluster
    :return: nothing
    Output: creates a csv file
    """
    df = pd.read_csv(datafile)
    df['clusters'] = kmeans.labels_
    df.to_csv("data/cleaned_data_with_clusters.csv")


if __name__ == "__main__":
    # data source
    datafile = "data/iqr_cleaned_data.csv"
    # load the data
    X = convert_csv_to_numpy_array(datafile)
    standardized = standardized_data(X)

    # perform kmeans analysis using sklearn.cluster import Kmeans
    kmeans = KMeans(n_clusters=3, random_state=0).fit(standardized)
    # plot the clusters
    plot_clusters_with_legend(standardized, kmeans)
    # print(kmeans.labels_)
