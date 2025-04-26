from kmeans import KMeans
from sklearn.datasets import make_blobs
from simplelinearregression import SimpleLinearRegression
import numpy as np

points = 1000

X = np.random.rand(points) * 10

noise = np.random.randn(points) * 10
y = 3 * X + 5 + noise  

points = []
for i in range (0, len(X)):
    points.append([X[i], y[i]])

slr = SimpleLinearRegression(points)

_, _, iteration_values, losses = slr.fit(learning_rate=0.01)

slr.draw_timelapse(iteration_values, losses, 0.00001)


'''
clusters = 12

standard_dev = 120.0
data, true_labels = make_blobs(n_samples=10000, centers=clusters, cluster_std=standard_dev, n_features=2, random_state=0)
dataframe = [] #[([], int)]
for point, label in zip(data, true_labels):
    tupled = (point, label)
    dataframe.append(tupled)

kmeans = KMeans(k_clusters=clusters)
a, b = kmeans.fit(standard_dev, dataframe)
'''




