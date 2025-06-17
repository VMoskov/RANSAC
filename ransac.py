import os 
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import random
import numpy as np
import warnings
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from criterions.mean_squared_error import MeanSquaredError, SquaredError
from models.linear_regressor import LinearRegressor
from dataset.points_dataset import PointsDataset, Point
from generators.line_generator import LineGenerator


class RANSAC:
    def __init__(self, model, criterion, loss, n_iterations=100000, threshold=1.0, sample_size=20, datapoints_required=20):
        self.model = model
        self.criterion = criterion  # criterion function to evaluate model predictions
        self.loss = loss   # loss function to evaluate model performance

        self.n_iterations = n_iterations
        self.threshold = threshold
        self.sample_size = sample_size
        self.datapoints_required = datapoints_required

        self.best_model = model


    def fit(self, data, verbose=False):
        self.best_inliers_mask = np.zeros(data.shape[0], dtype=bool)
        best_loss = float('inf')

        X, y = self._transform_data(data)
        n_samples = len(data)
        
        for _ in range(self.n_iterations):
            X_sample, y_sample = self._random_sample(X, y)

            model_candidate = self.model.fit(X_sample, y_sample)
            inliers_mask = self._get_inliers_mask(X, y, model_candidate)

            if inliers_mask.sum() > self.best_inliers_mask.sum() and inliers_mask.sum() >= self.datapoints_required:
                self.best_inliers_mask = inliers_mask
                # we don't calculate loss here, more inliers is always better

        if self.best_inliers_mask.any():
            X_final, y_final = X[self.best_inliers_mask], y[self.best_inliers_mask]

            self.best_model = self.model.fit(X_final, y_final)
            self.best_inliers = data[self.best_inliers_mask]

            if verbose: self._diagnostics(X, y)

        else:  # no consensus set found
            self.best_model = None
            self.best_inliers = None
            warnings.warn('No inliers found. The model may not fit the data well.')

        return self
    
    def predict(self, data):
        if self.best_model is None:
            raise RuntimeError('Model has not been fitted yet. Call fit() before predict().')
        
        X, _ = self._transform_data(data)
        return self.best_model.predict(X)

    def _random_sample(self, X, y):
        if len(X) < self.sample_size:
            raise ValueError('Not enough data points to sample from.')
        sample_indices = random.sample(range(len(X)), self.sample_size)
        return X[sample_indices], y[sample_indices]

    def _get_inliers_mask(self, X, y, model):
        preds = model.predict(X)
        losses = self.criterion(y, preds)
        return losses < self.threshold

    def _transform_data(self, data):
        numpy_data = data.to_numpy() if hasattr(data, 'to_numpy') else np.asarray(data)

        if numpy_data.ndim != 2 or numpy_data.shape[1] != 2:
            raise ValueError("Data must be convertible to an (N, 2) numpy array.")

        X_raw = numpy_data[:, :-1]
        y = numpy_data[:, -1]

        poly = PolynomialFeatures(degree=1, include_bias=True)
        X_transformed = poly.fit_transform(X_raw)
        
        return X_transformed, y
    
    def _diagnostics(self, X, y):
        X_inliers = X[self.best_inliers_mask]
        y_inliers = y[self.best_inliers_mask]
        
        outliers_mask = ~self.best_inliers_mask
        X_outliers = X[outliers_mask]
        y_outliers = y[outliers_mask]

        inliers_pred = self.best_model.predict(X_inliers)
        inliers_loss = self.loss(y_inliers, inliers_pred)

        outliers_pred = self.best_model.predict(X_outliers)
        outliers_loss = self.loss(y_outliers, outliers_pred)

        total_pred = self.best_model.predict(X)
        total_loss = self.loss(y, total_pred)

        print(f'Inliers Loss: {inliers_loss}, Outliers Loss: {outliers_loss}, Total Loss: {total_loss}')

    def visualize(self, data):
        if self.best_model is None:
            raise RuntimeError('Model has not been fitted yet. Call fit() before visualize().')
        
        numpy_data = data.to_numpy() if hasattr(data, 'to_numpy') else np.asarray(data)
        inliers = self.best_inliers.to_numpy() if self.best_inliers is not None else numpy_data[self.best_inliers_mask]

        outliers_mask = ~self.best_inliers_mask
        outliers = numpy_data[outliers_mask]

        x_min, x_max = numpy_data[:, 0].min(), numpy_data[:, 0].max()
        line_x = np.linspace(x_min, x_max, 100).reshape(-1, 1)

        poly = PolynomialFeatures(degree=1, include_bias=True)
        line_x_transformed = poly.fit_transform(line_x)
        line_y = self.best_model.predict(line_x_transformed)

        plt.figure(figsize=(10, 6))
        plt.scatter(outliers[:, 0], outliers[:, 1], color='silver', label='Outliers', s=20)
        plt.scatter(inliers[:, 0], inliers[:, 1], color='green', label='Inliers', s=40)
        plt.plot(line_x, line_y, color='red', label='Fitted Line')

        plt.title('RANSAC Regression')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
        

if __name__ == '__main__':
    generator = LineGenerator(
        num_samples=100,
        noise_level=0.3,
        x_range=(-10, 10),
        slope_range=(-20, 20),
        intercept_range=(-10, 10),
        jitter=0.05,
        salt_pepper_ratio=0.5
    )
    dataset = generator.generate()
    
    ransac = RANSAC(
        model=LinearRegressor(num_features=1),
        criterion=SquaredError(),
        loss=MeanSquaredError(),
        n_iterations=100000,
        threshold=0.05,
        sample_size=10,
        datapoints_required=10
    )

    ransac.fit(dataset, verbose=True)
    ransac.visualize(dataset)