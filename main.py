from kmeans import KMeans
from sklearn.datasets import make_blobs

clusters = 6

standard_dev = 20.0
data, true_labels = make_blobs(n_samples=3000, centers=clusters, cluster_std=standard_dev, n_features=2, random_state=0)
dataframe = [] #[([], int)]
for point, label in zip(data, true_labels):
    tupled = (point, label)
    dataframe.append(tupled)

kmeans = KMeans(k_clusters=clusters)
a, b = kmeans.fit(standard_dev, dataframe)

