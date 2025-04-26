import numpy as np
from matplotlib import pyplot as plt

class SimpleLinearRegression:
    def __init__(self, existing_points, lambda_reg=0.1):  
        self.existing_points = existing_points
        self.lambda_reg = lambda_reg 

    def update_thetas(self, theta0, theta1, theta0_cost_d, theta1_cost_d, learning_rate):
        damping_factor = 0.5
        theta0 -= damping_factor * learning_rate * theta0_cost_d
        theta1 -= damping_factor * learning_rate * theta1_cost_d
        return theta0, theta1

    def update_theta0(self, y_pred, y_actual, theta0):
        sum_error = np.sum(y_pred - y_actual)
        lasso_penalty = self.lambda_reg * np.sign(theta0)
        grad = sum_error * 2 / len(y_pred) + lasso_penalty
        return grad

    def update_theta1(self, y_pred, y_actual, x_actual, theta1):
        sum_error = np.sum((y_pred - y_actual) * x_actual)
        lasso_penalty = self.lambda_reg * np.sign(theta1)
        grad = sum_error * 2 / len(x_actual) + lasso_penalty
        return grad

    def loss_function(self, y_pred, y_actual, theta0, theta1):
        mse_loss = np.mean((y_pred - y_actual) ** 2)
        lasso_loss = self.lambda_reg * (abs(theta0) + abs(theta1))
        total_loss = mse_loss + lasso_loss
        return total_loss

    def fit(self, epsilon=0.01, learning_rate=0.001, max_iter=10000):
        iteration_values = []
        losses = []
        loss_difference = np.inf

        theta0 = np.random.uniform(-epsilon * 10, epsilon * 10)
        theta1 = np.random.uniform(-epsilon * 10, epsilon * 10)

        train_x = np.array([point[0] for point in self.existing_points])
        train_y = np.array([point[1] for point in self.existing_points])

        iteration = 0

        while loss_difference > epsilon and iteration < max_iter:
            predictions = theta0 + theta1 * train_x

            grad_theta0 = self.update_theta0(predictions, train_y, theta0)
            grad_theta1 = self.update_theta1(predictions, train_y, train_x, theta1)

            theta0, theta1 = self.update_thetas(
                theta0, theta1, grad_theta0, grad_theta1, learning_rate
            )

            loss = self.loss_function(predictions, train_y, theta0, theta1)
            losses.append(loss)

            if iteration > 0:
                loss_difference = abs(losses[-1] - losses[-2])

            iteration_values.append([theta0, theta1])

            print(f"Iteration {iteration}, Loss: {loss:.4f}, theta0: {theta0:.4f}, theta1: {theta1:.4f}")

            iteration += 1

        return theta0, theta1, iteration_values, losses

    def draw_timelapse(self, iteration_values, losses, pause_time=0.25):
        train_x = np.array([item[0] for item in self.existing_points])
        train_y = np.array([item[1] for item in self.existing_points])
        x_vals = np.linspace(min(train_x), max(train_x), 100)

        fig, ax = plt.subplots()
        
        for i in range(len(iteration_values)):
            ax.cla()  
            theta0, theta1 = iteration_values[i]
            
            y_vals = theta0 + theta1 * x_vals

            ax.plot(x_vals, y_vals, color='red', label='Regression Line')
            ax.scatter(train_x, train_y, color='green', label='Training Data')

            ax.set_ylim(0, max(train_y) + 1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f"Iteration {i+1}, Loss: {losses[i]:.4f}")
            ax.legend()
            
            plt.pause(pause_time)

        plt.show()
