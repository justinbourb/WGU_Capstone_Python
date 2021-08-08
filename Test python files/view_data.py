#%%



'''
Purpose: This file will allows viewing the properties
    of data files before cleaning is completed.
Input: a csv file to be inspected
Return: none
Output: file information
Lessons learned: Python vs Ipython
'''
# import packages
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib

plt.style.use('ggplot')

'''
% commands are "magic" commands for
With this backend, the output of plotting commands
is displayed inline within frontends like the
Jupyter notebook, directly below the code cell
that produced it. The resulting plots will then
also be stored in the notebook document.
'''

#%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None

#change the working directory
path = "C:\\git_local\\WGU_Capstone\\Cleaned data"
os.chdir(path)
# read the data
csv_data = pd.read_csv('working_data.csv')

# shape and data types of the data
print(csv_data.shape)
print(csv_data.dtypes)

# select numeric columns
csv_data_numeric = csv_data.select_dtypes(include=[np.number])
numeric_cols = csv_data_numeric.columns.values
print(numeric_cols)

# select non numeric columns
csv_data_non_numeric = csv_data.select_dtypes(exclude=[np.number])
non_numeric_cols = csv_data_non_numeric.columns.values
print(non_numeric_cols)

#create a heat map of missing data
cols = csv_data.columns[0:]
#blue is missing, yellow is not missing
colours = ['#000099', '#ffff00']
sns.heatmap(csv_data[cols].isnull(), cmap=sns.color_palette(colours))

# show % of missing data.
print("Percent of missing data per column:")
for col in csv_data.columns:
    pct_missing = np.mean(csv_data[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))

#%%

#creating a histogram of missing data
# create indicators for missing data
for col in csv_data.columns:
    missing = csv_data[col].isnull()
    num_missing = np.sum(missing)

    if num_missing > 0:
        print('created missing indicator for: {}'.format(col))
        csv_data['{}_ismissing'.format(col)] = missing


# then based on the indicator, plot the histogram of missing values
missing_columns = [col for col in csv_data.columns if 'ismissing' in col]
csv_data['num_missing'] = csv_data[missing_columns].sum(axis=1)

csv_data['num_missing'].value_counts().reset_index().sort_values(by='index').plot.bar(x='index', y='num_missing')

