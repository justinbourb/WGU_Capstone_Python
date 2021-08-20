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

# User’s Guide  
This project was built using the R and Python languages.  Python code was run with Anaconda Navigator 2.0.4 and PyCharm Professional 2020.2.1.  R analysis was done using Anaconda Navigator 2.0.4 and RStudio 1.1.456.  Both RStudio and PyCharm were run  from within Anaconda.

Further information can be found here:  
Anaconda: https://docs.anaconda.com/anaconda/navigator/  
PyCharm: https://www.jetbrains.com/pycharm/features/  
RStudio: https://www.rstudio.com/  

Opening RStudio in Anaconda requires creating a new environment and selecting Python and R.

To run the R file, open Rstudio from Anaconda and simply run the file.  Packages are installed automatically.

To run the remaining files requires Python.  Open PyCharm Professional from Anaconda. After starting the PyCharm application the project can be loaded by clicking File -> Open and selecting the directory where the project is stored.


Dependencies need to be installed before the project files will run.  Each file lists their dependencies at the top of the file via import statements.  There is also a requirements.txt file included with the project. PyCharm will notify the user of any missing dependencies and PyCharm will be able to install them when prompted to do so.  Both installation via “pip install” console command or via GUI is available in PyCharm.

Data analysis is done by running the individual Python files listed in the Python folder of the project.  These Python files require the csv data files stored in the data directory. The analysis provided by the files in the Python folder allowed me to build the dashboard app stored in the “Python/dash app” folder.

The dash app does not require any other Python file to run, just the data files.  Run the app.py file to load the dashboard server.

Then follow the link in any browser to view the dashboard.
