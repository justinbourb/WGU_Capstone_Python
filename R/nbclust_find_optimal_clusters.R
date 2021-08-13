" Purpose: Import cleaned data from the csv file and
            perform cluster analysis using NbClust package
  Input:  iqr_cleaned_data.csv was created using interquartile range analysis
          to remove outliers from the raw dataset.
  Returns: This script find the optimal number of clusters using the NbClust
          package and kmeans anaylsis.
  
  Lessons learned:
          1) importing csv files into R
          2) Ideal number of clusters analysis
          3) Dealing with memory limitations in R by using subsets
              of the data.
          4) accessing data frame by column and row range
"


iqr_cleaned_data <- read.csv("iqr_cleaned_data.csv")
install.packages("NbClust")
library(NbClust)
#complete NbClust analysis in three parts due to memory limitations
#0-34741, 34742-69482, 69483-104223
#This utilizes 57% of system memory
#104223 / 3 = 34741, 34741 * 2 = 69482
#Results: ~30 minutes run time, Error: cannot allocate 8.0 Gb
#Trying 1/3 again
#0-11580
#This utilizes 15% of system memory
#Results: ~2 hours run time, completed successfully
NbClust(data = iqr_cleaned_data$price[0:11580], diss = NULL, distance = "euclidean", min.nc = 2, max.nc = 15, 
        method = "kmeans", index = "all", alphaBeale = 0.1)





