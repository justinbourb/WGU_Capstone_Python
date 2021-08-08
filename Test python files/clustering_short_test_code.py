import csv

from sklearn.cluster import KMeans

import numpy as np
import pandas as pd

'''
source: https://stackoverflow.com/questions/28344660/how-to-identify-cluster-labels-in-kmeans-scikit-learn
'''


def load_data():
    datafile = "data/iqr_cleaned_data.csv"

    '''
    names:
    the column names are stored in names,
    numpy documentation says you can reference by name with the usecols variable
    but I couldn't get it to work
    usecols: 
    3,4 corresponds to price and item_id
    '''
    price_data = np.genfromtxt(
        datafile,
        delimiter=",",
        names="product_id, seller_id, price, item_id, customer_id, order_timestamp",
        usecols=2,
        skip_header=1
    )
    item_data = np.genfromtxt(
        datafile,
        delimiter=",",
        names="product_id, seller_id, price, item_id, customer_id, order_timestamp",
        usecols=range(3, 4),
        skip_header=1
    )
    '''
    delimiter: csv files use ,
    names: file headers, should be able to access via names but it wasn't working
    usecols: column range
    skip_headers:1 skips first line
    dtype: format type for numpy 'f8' = float64
        This type is required for kmeans.fit()
    deletechars: removes unwanted characters
    '''
    data = np.genfromtxt(
        datafile,
        delimiter=",",
        names="product_id, seller_id, price, item_id, customer_id, order_timestamp",
        usecols=range(2, 4),
        skip_header=1,
        dtype=[np.float64, np.float32],
        deletechars=', '
    )

    return data


def load_data_csv():
    datafile = "data/iqr_cleaned_data.csv"
    results = []
    with open(datafile) as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',')
        results = [row for row in csv_read]

    return results


def clustering_example():
    """
    Purpose: This function creates clusters from hard coded example data
    Returns: The clusters as labels array and the numpy array of the data as x_2d
    """
    x1 = [[1], [1], [2], [2], [2], [3], [3], [7], [7], [7]]
    x2 = [[1], [1], [2], [2], [2], [3], [3], [7], [7], [7]]

    x_2d = np.concatenate((x1, x2), axis=1)

    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
    labels = kmeans.fit(x_2d)
    return labels, x_2d


def clustering(data):
    """
    Purpose: This function creates clusters from my projects data
    Returns: The clusters as labels array and the numpy array of the data as x_2d
    """
    numpified = np.concatenate((data[:, 0], data[:, 1]), axis=1)
    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
    labels = kmeans.fit(numpified)
    return labels


def some_junk():
    '''
    Purpose: This function applies kmeans analysis to my data set.
        However, it only returns 3 clusters, each column is considered one item.
        Individual rows are not considered, which is not correct.
    '''
    data = load_data_csv()
    kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)

    npdata = np.array(data)
    print(len(npdata[1:, 5]))
    # this results in len=3, but should be 100,000
    numpified = (npdata[1:, 2], npdata[1:, 3], npdata[1:, 5])
    # requires the use of a pandas dataframe??////
    # try this tomorrow with pandas dataframememememem
    # https://blog.floydhub.com/introduction-to-k-means-clustering-in-python-with-scikit-learn/
    a = pd.to_datetime(['17-10-2010 07:15:30', '13-05-2011 08:20:35', "15-01-2013 09:09:09"])

    # this will iterate the third column (timestamp) and change the timestamp to an int
    # by removing any characters not a digit and changing the result to an int
    with np.nditer(numpified[2], op_flags=['readwrite']) as it:
        for x in it:
            astring = str(x)
            anint = int(''.join(c for c in astring if c.isdigit()))
            x[...] = anint

    '''
    time = '17-10-2010 07:15:30'
    #removes any characters not a digit
    print(int(''.join(c for c in time if c.isdigit())))
    '''
    # kmeans.fit(numpified)
    kmeans = kmeans.fit(numpified)
    print(kmeans.labels_)

def manual_kmeans():
    # source: https://blog.floydhub.com/introduction-to-k-means-clustering-in-python-with-scikit-learn/
    from sklearn.datasets import make_blobs

    # Generate 2D data points
    X, _ = make_blobs(n_samples=10, centers=3, n_features=2,
                      cluster_std=0.2, random_state=0)

    # Convert the data points into a pandas DataFrame
    import pandas as pd

    # Generate indicators for the data points
    obj_names = []
    for i in range(1, 11):
        obj = "Object " + str(i)
        obj_names.append(obj)

    # Create a pandas DataFrame with the names and (x, y) coordinates
    data = pd.DataFrame({
        'Object': obj_names,
        'X_value': X[:, 0],
        'Y_value': X[:, -1]
    })

    # Preview the data
    print('Initial data')
    print(data.head())

    # Initialize the centroids
    # centriods are used to calculate the distance from a point to the center of a cluster (centroid)
    c1 = (-1, 4)
    c2 = (-0.2, 1.5)
    c3 = (2, 2.5)

    # A helper function to calculate the Euclidean diatance between the data
    # points and the centroids

    def calculate_distance(centroid, X, Y):
        distances = []

        # Unpack the x and y coordinates of the centroid
        c_x, c_y = centroid

        # Iterate over the data points and calculate the distance using the
        # given the Euclidean distance formula
        '''
        The zip() function returns a zip object, which is an iterator of tuples where the first item in each passed 
        iterator is paired together, and then the second item in each passed iterator are paired together etc.

        If the passed iterators have different lengths, the iterator with the least items decides the length of the new 
        iterator.
        '''
        for x, y in list(zip(X, Y)):
            root_diff_x = (x - c_x) ** 2
            root_diff_y = (y - c_y) ** 2
            distance = np.sqrt(root_diff_x + root_diff_y)
            distances.append(distance)

        return distances

    # Calculate the distance and assign them to the DataFrame
    # The distance to each centroid is calculated for each data point to find the smallest distance
    # The smallest distance = the cluster assignment
    data['C1_Distance'] = calculate_distance(c1, data.X_value, data.Y_value)
    data['C2_Distance'] = calculate_distance(c2, data.X_value, data.Y_value)
    data['C3_Distance'] = calculate_distance(c3, data.X_value, data.Y_value)

    # Preview the
    print('\n Data with distance to centroids')
    print(data.head())

    # Get the minimum distance centroids
    # argmin Returns the indices of the minimum values along an axis.
    # This step doubles the # of rows, but changes the x, y values somehow... why??

    data['Cluster'] = data[['C1_Distance', 'C2_Distance', 'C3_Distance']].apply(np.argmin, axis=1)

    print("\n data.Cluster output")
    print(data.Cluster)
    # Map the centroids accordingly and rename them
    # data['Cluster'] = data['Cluster'].map({'C1_Distance': 'C1', 'C2_Distance': 'C2', 'C3_Distance': 'C3'})

    # remove doubled rows created by argmin
    # super hacky solution, but I'm not sure why the rows are doubling
    data = data[:5]
    # Get a preview of the data
    print('\n Data with cluster assignments')
    print(data.head(10))

def kmeans_example():
    # source: https://blog.floydhub.com/introduction-to-k-means-clustering-in-python-with-scikit-learn/
    # Using scikit-learn to perform K-Means clustering
    # from sklearn.cluster import Kmeans
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # Generate 2D data points
    X, _ = make_blobs(n_samples=10, centers=3, n_features=2,
                      cluster_std=0.2, random_state=0)
    '''
    The data input to Kmeans.fit() should fit the following format:
    [
    [float float],
    [float float],
    [float float],
    ]
    0) The example uses an numpy.ndarray
    1) The data should be inside a list with a space in between each data point.
    2) Data should be "normalized" (what's the real term here?) so they fit the format #.#####
        Nothing more than the single's digit is allowed or the results will be skewed by large numbers.

    Questions:
    How do I apply the kmeans clusters back to the original csv file as a new column?????????
    '''
    print(type(X))
    # Specify the number of clusters (3) and fit the data X
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

    # Preview the data
    print('Initial data')
    print(X)

    # Get the cluster centroids
    print('\n  kmeans.cluster_centers_ aka centroids')
    print(kmeans.cluster_centers_)

    # Get the cluster labels
    print('\n kmeans.labels_ aka cluster assignments for each row')
    print(kmeans.labels_)

    def plot_clusters(X):
        # Plotting the cluster centers and the data points on a 2D plane
        plt.scatter(X[:, 0], X[:, -1])

        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x')

        plt.title('Data points and cluster centroids')
        plt.show()

    plot_clusters(X)

def convert_csv_to_numpy_array(file):
    '''
    Purpose: This function:
                1) loads a csv using pandas
                2) converts it to a pandas data frame
                3) converts it to a numpy array and returns the array
            A numpy array is required for sklean.cluster Kmeans.fit()
    :param file:
    :return a numpy array:
    '''
    # loads the file, loads columns 2,3 only (price and item id), header row = 0 is not loaded
    data_set = pd.read_csv(file, usecols=[2,3], header=0)
    data_frames = pd.DataFrame(data_set)

    required_array = np.array(data_frames.values)
    return required_array

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    datafile = "data/iqr_cleaned_data.csv"
    X = convert_csv_to_numpy_array(datafile)
    print(X)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    def plot_clusters(X):
        # Plotting the cluster centers and the data points on a 2D plane
        plt.scatter(X[:, 0], X[:, -1])

        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x')

        plt.title('Data points and cluster centroids')
        plt.xlabel('Item price (price)')
        plt.ylabel('# of items ordered (item_id)')
        plt.show()

    plot_clusters(X)
