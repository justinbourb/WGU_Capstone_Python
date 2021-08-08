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

from pycaret.datasets import get_data
import pandas

if __name__ == "__main__":
    file = 'data/cleaned_data_with_clusters.csv'
    # pycaret's get_data results in a an error
    #data = get_data('data/cleaned_data_with_clusters.csv')
    #data_set = pandas.read_csv(file, header=0)

    import os
    print(os.path.expanduser('~'))