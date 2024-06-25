'''
This a model of kmeans clustering. The data is sourced from the repo.
Clustering is done for the Mall_Customers dataset (given in the internship task page from Kaggle).
Clustering is based on Spending Score and Income Range.
'''
#importing necessary libraries
import pandas as pd #read csv files
import numpy as np
import matplotlib.pyplot as plt #plotting 
from sklearn.cluster import KMeans as km #k-means cluster
#reading and organizing the data

link = "https://github.com/Vevelugu/PRODIGY_ML_02/blob/main/Mall_Customers.csv?raw=true"
data = pd.read_csv(link)
spend_habit = data['Spending Score (1-100)']
income = data['Annual Income (k$)']
incomeXsh = np.column_stack((income, spend_habit))



#Kmeans clustering with 5 centers and plotting the graph
km_res = km(n_clusters=5).fit(incomeXsh)
plt.scatter(incomeXsh[:,0], incomeXsh[:,1], alpha=0.5, c=km_res.labels_.astype(float))
plt.xlabel("Income Range")
plt.ylabel("Spending Habit")
plt.title("Income vs Spending Habit Clustering")
plt.show()

