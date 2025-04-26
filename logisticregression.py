import numpy as np
from matplotlib import pyplot as plt

class LogisticRegression:
    def __init__(self, X, labels):
        self.X = np.c_[np.ones(X.shape[0]), X]
        self.labels = labels
        self.loss_history = []
        self.theta = np.zeros(X.shape[1] + 1)
        self.theta_history = [self.theta]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def calculate_loss(self):
        m = len(self.labels)
        predictions = self.sigmoid(np.dot(self.X, self.theta))
        loss = -(1/m) * np.sum(self.labels * np.log(predictions) + (1 - self.labels) * np.log(1 - predictions))
        self.loss_history.append(loss)

    def update(self, learning_rate = 0.01):
        m = len(self.labels)
        predictions = self.sigmoid(np.dot(self.X, self.theta))
        self.theta = np.subtract(self.theta, learning_rate * 1/m * np.dot(self.X.T, (np.subtract(predictions, self.labels))))
        self.theta_history.append(self.theta)
        
    def fit(self, max_iter = 100, tolerance = 0.01):
        loss_difference = np.inf
        iterations = 0
        while loss_difference > tolerance and iterations < max_iter:
            self.calculate_loss()
            self.update()
            if (iterations > 1):
                loss_difference = self.loss_history[iterations - 1] - self.loss_history[iterations - 2]
            iterations += 1