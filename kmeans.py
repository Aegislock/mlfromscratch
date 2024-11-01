import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sklearn.datasets import make_blobs

def calculate_euclidean_distance(point1, point2):
    distance = 0
    for coord1, coord2 in zip(point1, point2):
        distance += pow(coord1 - coord2, 2)
    return math.sqrt(distance)

class KMeans():
    def __init__(self, k_clusters, max_iter=1000, tolerance=1e-4):
        self.k_clusters = k_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.colors = [
            'red', 
            'blue', 
            'green', 
            'yellow', 
            'orange', 
            'purple', 
            'cyan', 
            'magenta', 
            'pink', 
            'brown', 
            'lime', 
            'navy', 
            'teal', 
            'gray', 
            'gold', 
            'silver'
        ]
    
    def fit(self, std, data):  
        centroids = self.initialize_centroids(data)
        clusters = {i: [] for i in range(self.k_clusters)}
        #Main Loop
        for i in range(0, self.max_iter):
            self.draw_timelapse(centroids, clusters, i, std=std)
            new_clusters = {}
            for key in clusters:
                new_clusters[key] = []
            for point in data:
                centroid_index = None
                min_distance = np.inf  # Initialize min_distance to infinity
                # Euclidean distance to each centroid
                for j in range(0, len(centroids)):
                    distance = np.linalg.norm(point[0] - centroids[j])
                    if distance < min_distance:
                        min_distance = distance
                        centroid_index = j
                new_clusters[centroid_index].append(point[0])
            new_centroids = self.calculate_centroids(new_clusters)
            count = 0
            for j in range(0, len(new_centroids)):
                if self.calculate_euclidean_distance(new_centroids[j], centroids[j]) < self.tolerance:
                    count += 1
            if count == len(centroids):
                return new_centroids, new_clusters
            else:
                centroids = new_centroids
                clusters = new_clusters
        return centroids, clusters

    def initialize_centroids(self, data):
        points = []
        for point, _ in data:
            points.append(point)
        centroids_indices = random.sample(range(len(points)), self.k_clusters)
        centroids = []
        for index in centroids_indices:
            centroids.append(points[index])
        return centroids

    #Error with this method where it returns empty
    def calculate_centroids(self, clusters):
        new_centroids = []
        for i in range(len(clusters)):  # Changed range syntax for clarity
            coord_sum = {k: 0.0 for k in range(len(clusters[i][0]))}
            new_centroid = []
            
            if len(clusters[i]) == 0:
                new_centroid = [0.0] * len(coord_sum)
            else:
                for j in range(len(clusters[i])):  # Changed range syntax for clarity
                    for k in range(len(clusters[i][j])):  # Changed range syntax for clarity
                        coord_sum[k] += clusters[i][j][k]
                # Changed this part to use list comprehension
                new_centroid = [coord_sum[k] / len(clusters[i]) for k in range(len(coord_sum))]  # Optimized

            # Added this line to append the new centroid to the list
            new_centroids.append(new_centroid)  
            
        return new_centroids  # This now returns the new centroids


    def draw_timelapse(self, centroids, clusters, iteration, std, pause_time=0.25):
        # Clear the current figure
        plt.clf()

        # Plot each cluster
        for i, cluster_points in clusters.items():
            cluster_color = self.colors[i]  # Get color for the current cluster
            if cluster_points:  # Ensure the cluster is not empty
                plt.scatter(
                    [point[0] for point in cluster_points],
                    [point[1] for point in cluster_points],
                    c=cluster_color,
                    marker='o',
                    s=80,
                    edgecolor='k',
                    label=f'Cluster {i}'
                )
        
        # Plot centroids
        for i, centroid in enumerate(centroids):
            plt.scatter(
                centroid[0],
                centroid[1],
                c=self.colors[i],
                marker='X',
                s=300,
                edgecolor='k',
                linewidths=2,
                label=f'Centroid {i}'
            )
        
        # Add labels and title
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'K-means Clustering Visualization - Iteration {iteration + 1}')
        plt.legend()
        
        plt.xlim(-3*1.6*std, 3*1.6*std)  # Adjust these limits as per your data
        plt.ylim(-3*std, 3*std)

        plt.show(block=False)  # Non-blocking show

        plt.pause(pause_time)  # Pause to create a timelapse effect
    
    def calculate_euclidean_distance(self, point1, point2):
        distance = 0
        for coord1, coord2 in zip(point1, point2):
            distance += pow(coord1 - coord2, 2)
        return math.sqrt(distance)
    
    def on_key(self, event):
        if event.key == 'q':  # Check if the pressed key is 'q'
            plt.close()

