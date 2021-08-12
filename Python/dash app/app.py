"""
Purpose: This file will create a dash app server.  This server displays the business analysis tool
    created for my WGU Capstone project.  It displays various graphs, charts and tree diagrams
    related to clustering customers based on their past purchase histories.
"""

# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import base64
import os
import pathlib

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import dash_auth


def split_product_and_price(data):
    """
    Purpose: This function accepts a pandas dataframe and returns a dataframe containing top 20 products and their
      prices. This function is hardcoded and not reusable without modification.
    :return: a pandas dataframe
    """
    extra_sorting_step = data.product_id.value_counts().sort_index(ascending=False).to_frame()
    sorted_items = extra_sorting_step.sort_values(by='product_id', ascending=False).head(20)
    names = []
    count = []
    for index, row in sorted_items.iterrows():
        all_items = str(row).split(' ')

        for item in all_items:
            if len(item) > 10:
                names.append(item.strip(','))
            try:
                int(item.strip("\\nName:"))
                count.append(item[0:3])
            except:
                # ignore any items which cannot be converted to int
                pass
    df = pd.DataFrame({'product_id': names})
    df['count'] = count
    return df


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# allowed user name and password combinations
LOGIN = {'test': 'test'}
# use dash basic authorization, advanced dash login requires a paid subscription or use of flask.
auth = dash_auth.BasicAuth(app, LOGIN)

# get the working directory
current_dir = os.getcwd()
# only change the working directory if we are in the dash app folder.  /data is the desired directory.
if current_dir.split('\\')[-1] == 'dash app':
    os.chdir('../../data')
# read the data
cleaned_data = pd.read_csv('cleaned_data_with_clusters.csv')
# an overly complicated method of sorting data follows below:
top_twenty_items = split_product_and_price(cleaned_data)

# dash requires static images to be formatted in base64
# clusters scatter plot
cluster_image_filename = '..\\Images\\clustering_scatter_plot.png'
cluster_encoded_image = base64.b64encode(open(cluster_image_filename, 'rb').read())
# decision tree
tree_image_filename = '..\\Images\\my_decisiontree.png'
tree_encoded_image = base64.b64encode(open(tree_image_filename, 'rb').read())

# create a histogram of the data
fig_hist_price = px.histogram(cleaned_data, x="price", nbins=100, title="Dataset price Histogram")
fig_hist_item = px.histogram(cleaned_data, x="item_id", color="item_id", nbins=100, title="Dataset item_id Histogram")

# create a bar graph
fig_bar_top_items = px.bar(top_twenty_items, x="product_id", y="count", color="product_id")
fig_bar_top_items['layout']['yaxis']['autorange'] = "reversed"

# create the layout for the app using pseudo html
app.layout = html.Div(children=[
    html.H1(children='Hello Readers'),

    html.Div(children='''
        A business intelligence tool.
    '''),
    # add a histogram
    html.Div(dcc.Graph(
        id='hist_graph_price',
        figure=fig_hist_price
    )),
    # add a histogram
    html.Div(dcc.Graph(
        id='hist_graph_item',
        figure=fig_hist_item
    )),
    # add a bar graph
    html.Div(dcc.Graph(
        id='bar_graph_top_twenty',
        figure=fig_bar_top_items,
    )),
    # add a table with pagination
    html.Div(dash_table.DataTable(
        id='datatable-paging',
        columns=[
            {"name": i, "id": i} for i in sorted(cleaned_data.columns)
        ],
        page_current=0,
        page_size=20,
        page_action='custom'
    )
    ),
    # source and decode the image of the cluster scatter plot
    html.Div(html.Img(src='data:image/png;base64,{}'
                      .format(cluster_encoded_image.decode()))),
    # source and decode the image of the decision tree
    html.Div(html.Img(src='data:image/png;base64,{}'
                      .format(tree_encoded_image.decode())))
])


# This call back handles paginating the table data
@app.callback(
    Output('datatable-paging', 'data'),
    Input('datatable-paging', "page_current"),
    Input('datatable-paging', "page_size"))
def update_table(page_current, page_size):
    return cleaned_data.iloc[
           page_current * page_size:(page_current + 1) * page_size
           ].to_dict('records')


# run the server
if __name__ == '__main__':
    app.run_server(debug=True)
