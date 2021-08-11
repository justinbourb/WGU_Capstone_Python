"""
Purpose: This file will allows viewing the properties
of data files before cleaning is completed.
Input: a csv file to be inspected
Return: none
Output: file information
Lessons learned:
    1) Python vs Ipython
    2) panda plots need their own cell or they overlap (print two plots on same graph)
    3) Jupyter is like python but worse, each step needs it's own cell
    4) Ipython is like python but worse, magic words create obfuscation
"""

import os
import csv
import pandas as pd

if __name__ == "__main__":
    path = "C:\\git_local\\WGU_Capstone_Python\\data"
    os.chdir(path)
    # read the data
    csv_data = pd.read_csv('working_data.csv')
    print('Original length of working data ' + str(len(csv_data)))
    # calculate Q1, Q3 and IQR
    Q1 = csv_data['price'].quantile(0.25)
    Q3 = csv_data['price'].quantile(0.75)
    IQR = Q3 - Q1

    print(len(csv_data))

    # This data set only has outliers higher than Q3
    # Find the index of all high outliers
    high_outliers = []
    for index, value in enumerate(csv_data['price']):
        if float(value) > (Q3 + 1.5 * IQR):
            high_outliers.append(index)

    # remove the row for all high outliers from the dataset
    for i in high_outliers:
        csv_data.drop([i], inplace=True)
    print(len(csv_data))

    # 112650 is the number of rows before outlier removal
    # 104223 is the number of rows after IQR outlier removal
    # This is a 7.480692 percent reduction in the number of rows

    # write iqr cleaned data to a csv
    with open('C:\\git_local\\WGU_Capstone_Python\\data\\iqr_cleaned_data.csv', 'w', encoding='UTF8',
              newline='') as file:
        writer = csv.writer(file)
        writer.writeheader()
        writer.writerows(csv_data)