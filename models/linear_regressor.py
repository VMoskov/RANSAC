import numpy as np


class LinearRegressor:
    def __init__(self, num_features):
        self.W = np.random.randn(num_features + 1)  # +1 for bias term

    def predict(self, X):
        # X is augmented
        return X @ self.W

    def fit(self, X, y):
        # X is augmented
        self.W = np.linalg.inv(X.T @ X) @ X.T @ y
        return self