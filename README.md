# WGU_Capstone_Python

My project is to analyze past online purchases to create a business intelligence tool.  My analysis follows the CRISP-DM methodology and includes data cleaning/preprocesing,
clustering, a decision tree to assign future customers to clusters and association rules mining to determine if any products are likely to be bought together.


The working_dataset.csv is based upon the Brazilian E-Commerce Public Dataset.  Working_data.csv was created using JOIN in MySQL to combine
the needed data from multiple tables in the Brazlian dataset.  
Brazilian E-Commerce Public Dataset by Olist is licensed under CC BY-NC-SA 4.0.  
Source: https://www.kaggle.com/olistbr/brazilian-ecommerce?select=olist_order_items_dataset.csv  
Owner: https://olist.com/  
License: https://creativecommons.org/licenses/by-nc-sa/4.0/  
  
Lessons learned:  
- Python vs Ipython
- Working with Juypter
- KMeans analysis with Python
- Working with numpy, pandas and sklearn and how they interact
- The importance of standardizing data before clustering
    - Large values skew clustering, all values should be similar (standardized) for best results
- Plotting in Python
- Creating Data trees in python
- Working with Anaconda
    - Don't mix up Anaconda Pycharm and Pycharm, it messes up the libraries
