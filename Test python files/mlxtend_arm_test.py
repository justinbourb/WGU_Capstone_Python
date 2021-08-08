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
# source: https://pbpython.com/market-basket-analysis.html


import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import os


def load_data(file):
    """
    Purpose: This function sets the working directory and loads the data in a pandas dataframe
    :return: a pandas dataframe
    """
    # set working directory and csv file path
    path = "C:\\git_local\\WGU_Capstone_Python"
    os.chdir(path)
    file = file
    return pd.read_csv(file)


def clean_data(df):
    """
    Purpose: some of the descriptions have spaces that need to be removed. We’ll also drop the rows that don’t have
    invoice numbers and remove the credit transactions (those with invoice numbers containing C).
    :param df:
    :return: cleaned df
    """
    df['Description'] = df['Description'].str.strip()
    df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
    df['InvoiceNo'] = df['InvoiceNo'].astype('str')
    return df[~df['InvoiceNo'].str.contains('C')]


def calculate_rules(df):
    # create frequent_items
    frequent_items = apriori(df, min_support=0.07, use_colnames=True)
    rules = association_rules(frequent_items, metric="lift", min_threshold=1)
    pd.set_option('display.max_columns', None)
    print(rules.head())
    # ValueError: The allowed values for a DataFrame are True, False, 0, 1.

def one_hot_encoding(df):
    """
    Purpose: we need to consolidate the items into 1 transaction per row with each product 1 hot encoded.
        For brevity, only data from France will be included
    :param df:
    :return:
    """
    #This would be equivalent to customer_id['item_id'].set_index('customer_id') for my dataset
    basket = (df[df['Country'] =="France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))
    def encode_units(x):
        """
        Set anything less than 0 to 0, anything greater to 1
        This is required for one hot encoding and mlxtend implementation of apriori
        :param x:
        :return:
        """
        if x <= 0:
            return 0
        if x >= 1:
            return 1

    basket_sets = basket.applymap(encode_units)
    basket_sets.drop('POSTAGE', inplace=True, axis=1)
    return basket_sets
if __name__ == "__main__":
    # load data
    # df = load_data('data\\cleaned_data_with_clusters.csv')
    df = load_data("data/test_data_mlxtend.csv")
    df = clean_data(df)
    basket_sets = one_hot_encoding(df)
    calculate_rules(basket_sets)
