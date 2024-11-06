## inspiration taken from various sources covering this well known example. 
## however, code is edited to add simplicity and comments added to show understanding of the core concepts regarding k-means clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

## transfers the .csv dataset into a pandas dataframe
df = pd.read_csv("/your/path/to/Mall_Customers.csv")


## isolates the values that will be  tested --> customers annual income and spending score
x = df.iloc[:, [3, 4]].values

## creates a list (wcss) that will store the cluster values 1-10
## creates a 'KMeans' object from the sklearn.cluster module (kmeans = KMeans()). 'n_clusters=i' sets the number of clusters to value 'i'. "init='k-means++'" sets the clusters farther apart, ensuring that as many datapoints as possible are covered, discouraging competition between centroids. 'random_state=42' directs the model to start with the same amount of random numbers each time it runs. 
## 'for i in range(1,11)' loops through number of clusters 'i' 10 times.
wcss = [KMeans(n_clusters=i, init='k-means++', random_state=42).fit(x).inertia_ for i in range(1, 11)]

## creates a 'KMeans' object from the sklearn.cluster module (kmeans = KMeans()). sets the number of clusters = 5. 
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
## 'kmeans.fit_predict(x)' trains the model on the values in variable 'x', finds the optimal locations for cluster centers based on the datapoints. then it assigns each datapoint to a cluster. 
y_kmeans = kmeans.fit_predict(x)

## sets the size of the graph to 8 by 8, making it square
plt.figure(figsize=(8, 8))
## defines the colors used for datapoints
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
## loops through the number of clusters set previously
for i in range(5):
    ## 'plt.scatter' creates a scatterplot of datapoints for each cluster
    ## 'x[y_kmeans == i, 0]' selects all datapoints that correspond to the value of i, then x-coordinates of the datapoints is extracted
    ## 'x[y_kmeans == i, 1]' selects 'y' coordinates for datapoints in cluster 'i'
    ## 's=100' sets the datapoint size
    ## 'c=colors[i]' sets the color equal to its corresponding cluster value
    ## "label=f'Cluster {i+1}'" provides a label for the clusters in the graph
    plt.scatter(x[y_kmeans == i, 0], x[y_kmeans == i, 1], s=100, c=colors[i], label=f'Cluster {i+1}')

## plots the centroids in the center of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
