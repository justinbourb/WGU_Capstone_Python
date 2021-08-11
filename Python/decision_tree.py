"""
Purpose: This file will create a decision tree.
        The input data is price and order_id (# of items purchased), the target data is cluster.
Input: a csv file to be inspected
Return: none
Output: an image of the decision tree
Lessons learned:
    1) Creating Data trees in python
    2) Interpreting Python data trees
"""

# source: https://www.w3schools.com/python/python_ml_decision_tree.asp
import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg


if __name__ == "__main__":
    file = "data/cleaned_data_with_clusters.csv"
    # loads the file, loads columns 2, 3, 6 only (price, item id, cluster), header row = 0 is not loaded
    data_set = pandas.read_csv(file, usecols=[3, 4, 7], header=0)

    # X = Data features (input)
    features = ['price', 'item_id']
    X = data_set[features]
    # Y = Values we want to predict (target)
    Y = data_set['clusters']

    # create a decision tree
    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X, Y)
    data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
    # create the graph
    graph = pydotplus.graph_from_dot_data(data)
    # save the graph as a png file
    graph.write_png('Images\\mydecisiontree.png')
    # read the saved image and plot the graph
    img = pltimg.imread('Images\\mydecisiontree.png')
    imgplot = plt.imshow(img)
    plt.show()

'''Results explained:

price | item_id <= 106.4 means that any price | item_id of 106.4 or lower will follow the True arrow (to the left), and the 
rest will follow the False arrow (to the right).

gini = 0.461 refers to the quality of the split, and is always a number between 0.0 and 0.5, where 0.0 would mean all 
of the samples got the same result, and 0.5 would mean that the split is done exactly in the middle.

samples = # of samples left at this point in the decision

value = [70375, 29836, 4012] represents the number of rows in [cluster 0, cluster 1, cluster 2]

'''