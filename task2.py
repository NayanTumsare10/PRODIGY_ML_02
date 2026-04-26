import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

X = data[["Annual Income (k$)", "Spending Score (1-100)"]].values

kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(X)

plt.figure(figsize=(6,5))

colors = ['red', 'blue', 'green', 'purple', 'orange']

for i in range(5):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], 
                color=colors[i], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1],
            color='black', s=120, label='Centroids')

plt.xlabel("Income")
plt.ylabel("Score")
plt.title("Customer Segmentation")

plt.legend()
plt.show()