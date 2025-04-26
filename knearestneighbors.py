import numpy as np
import random
import math

def calculate_euclidean_distance(point1, point2):
    distance = 0
    for coord1, coord2 in zip(point1, point2):
        distance += pow(coord1 - coord2, 2)
    return math.sqrt(distance)

class KNearestNeighbors():
    def __init__(self, k, existing_points):
        self.k = k
        #Existing Points ([x, y, z...], Class)
        self.existing_points = existing_points

    def fit(self, data):
        fitted = []
        for point in data:
            distances = []
            for existing_point in self.existing_points:
                distances.append((calculate_euclidean_distance(point, existing_point[0]), existing_point))
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            frequency = {point: 0 for point in k_nearest}
            for point in k_nearest:
                frequency[point[1][1]] += 1
            most = 0
            mostClass = None
            for key in frequency:
                if frequency[key] > most:
                    most = frequency[key]
                    mostClass = key
            fitted.append((point, mostClass))  
        return fitted
        
