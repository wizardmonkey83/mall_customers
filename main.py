## clustering planatary systems based on:
## Stellar Effective Temperature [K]
## Stellar Radius [Solar Radius]
## Stellar Mass [Solar mass]
## Stellar Metallicity [dex]
## Planet Radius [Jupiter Radius / Earth Radius]
## Planet Mass [Earth Mass / Jupiter Mass]
## Eccentricity
## Number of Planets
## Discovery Year

import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans


## loads the .csv file to a pandas dataframe
df = pd.read_csv('/content/planatary_clustering.csv', sep=',', on_bad_lines='skip')

## identifies the values that the model will track
NUMERIC_COLUMNS = ['st_teff', 'st_rad', 'st_mass', 'st_met', 'pl_radj', 'pl_bmassj', 'pl_orbeccen', 'sy_pnum', 'disc_year']

## loops through each value in 'NUMERIC_COLUMNS'
for column in NUMERIC_COLUMNS:
    ## converts all values to numeric values, forcing any non-numerical values to NaN
    df[column] = pd.to_numeric(df[column], errors='coerce')

## since this dataset isn't perfect, a lot of values are missing, this drops the missing values. another route could be taking the mean of all values in a column and replacing a missing value with the columns mean. 
df.dropna(subset=NUMERIC_COLUMNS, inplace=True)

## 'x' is defined by the numerical values of names in 'NUMERICAL_COLUMNS'. this extracts the values that will be used for clustering
x = df[NUMERIC_COLUMNS].values

## this is necessary for finding the optimal cluster count
## uses matplotlib to create a plot, with x-axis = number of clusters 10, the y-axis holding wcss values, and "marker='o'" making each datapoint a circle
wcss = [KMeans(n_clusters=i, init='k-means++', random_state=42).fit(x).inertia_ for i in range(1, 11)]

## didn't feel the need to add in the code for graphinh. optimal cluster size is 3

## creates a 'KMeans' object from the sklearn.cluster module (kmeans = KMeans()). sets the number of clusters = 5. 
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
## 'kmeans.fit_predict(x)' trains the model on the values in variable 'x', finds the optimal locations for cluster centers based on the datapoints. then it assigns each datapoint to a cluster. 
y_kmeans = kmeans.fit_predict(x)

## adds the cluster labels to the dataframe for visualization
df['Cluster'] = y_kmeans

## colors that will represent the clusters
colors = ['pink', 'teal', 'orange']

## generates a pair plot for for the clusters to determine which relations are best for clustering/finding differences
sns.pairplot(df[NUMERIC_COLUMNS + ['Cluster']], hue='Cluster', palette=colors[:3])
plt.show()
