#importing necessary libraries
import pandas as pd #read csv files
import numpy as np
import matplotlib.pyplot as plt #plotting 
from sklearn.cluster import KMeans as km #k-means cluster
from sklearn import metrics
from scipy.spatial.distance import cdist

#reading and organizing the data
data = pd.read_csv(r"D:\AK\Career\Python\ProdigyInfotech\Task02_kmeansclustering\archive\Mall_Customers.csv")
age = data['Age']
spend_habit = data['Spending Score (1-100)']
agexspendhabit = np.column_stack((age, spend_habit))
print(agexspendhabit)
income = data['Annual Income (k$)']
print(income)
incomeXsh = np.column_stack((income, spend_habit))
#find optimal number of clusters using elbow method and distortion
def distortion_method(datfram):
    distortions = []
    mappings = {}
    N = range(1,10)
    for n in N:
        km_model = km(n_clusters=n).fit(datfram)
        distorter = sum(np.min(cdist(datfram, km_model.cluster_centers_, 'euclidean'), axis = 1))/datfram.shape[0]
        distortions.append(distorter)
        mappings[n] = distorter

    plt.plot(N, distortions, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('Elbow Method using distortion')
    plt.show()

def inertia_method(datfram):
    inertias = []
    mappings = {}
    N = range(1,10)
    for n in N:
        km_model = km(n_clusters=n).fit(datfram)
        inertias.append(km_model.inertia_)
        mappings = km_model.inertia_
    
    plt.plot(N, inertias, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Inertias')
    plt.title('Elbow method using Inertia')
    plt.show()   
    
a = distortion_method(agexspendhabit)
b = inertia_method(agexspendhabit)

#now for actual clustering after asking user for input of number of clusters
k = input("Enter a value for number of clusters needed(1 to 10)")
if k not in range(1,10):
    k = input("Enter a value between 1 and 10")
k = int(k)
km_res = km(n_clusters=k).fit(incomeXsh)
plt.scatter(incomeXsh[:,0], incomeXsh[:,1], alpha=0.25, c=km_res.labels_.astype(float))
plt.xlabel("Income Range")
plt.ylabel("Spending Habit")
plt.title("Income vs Spending Habit Clustering")
plt.show()


