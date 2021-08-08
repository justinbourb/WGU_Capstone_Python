"""
Purpose: This file will mine the association rules form each cluster.
Input: a csv file to be inspected
Return: none
Output:
Lessons learned:
    1) Don't mix up Anaconda Pycharm and Pycharm, it messes up the libraries
    2) The data should be split into separate tables or data frames based on
        cluster assignment before processing.  The ARM algorithm does not consider cluster,
        so it should be run on each cluster separately.
"""
# source: https://www.pycaret.org/tutorials/html/ARUL101.html
from pycaret.arules import *
import pandas
import os


def load_data():
    """
    Purpose: This function sets the working directory and loads the data in a pandas dataframe
    :return: a pandas dataframe
    """
    # set working directory and csv file path
    path = "C:\\git_local\\WGU_Capstone_Python"
    os.chdir(path)
    file = 'data\\cleaned_data_with_clusters.csv'
    # pycaret's get_data results in a an error, pycaret also accepts pandas data frame
    # I think in this case we want to include the headers as pycaret setup() uses headers
    return pandas.read_csv(file)


def setup_model(data):
    """
        The setup() function initializes the environment in pycaret and transforms the transactional dataset into
        a shape that is acceptable to the apriori algorithm. It requires two mandatory parameters: transaction_id
        which is the name of the column representing the transaction id and will be used to pivot the matrix,
        and item_id which is the name of the column used for the creation of rules. Normally, this will be the
        variable of interest. You can also pass the optional parameter ignore_items to ignore certain values when
        creating rules.

        In my case the transaction_id would equal customer_id and the item_id would equal product_id.
        customer_id can identify combined purchases and product_id identifies the product.
    """
    setup0 = setup(
        data=data,
        transaction_id='customer_id',
        item_id='product_id'
    )
    # create the model (association rules)
    # this step is very problematic, and crashes if dataset is too large
    # shouldn't machine learning libraries handle large data by default?
    # only works in juypter / ipython????
    model = create_model()


if __name__ == "__main__":
    data_set = load_data()
    # split the dataset into subsets based on cluster
    # cluster0, cluster1, cluster2 = data_set.groupby(data_set['clusters'])
    cluster0 = data_set[data_set['clusters'] == 0]
    cluster1 = data_set[data_set['clusters'] == 1]
    cluster2 = data_set[data_set['clusters'] == 2]
    #setup_model(cluster1)
    print('c1: ' + str(len(cluster1)))
    print('c2: ' + str(len(cluster2)))
    print(cluster2['customer_id'])
    print(cluster2['product_id'])