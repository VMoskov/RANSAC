import random
import numpy as np
import warnings
from sklearn.preprocessing import PolynomialFeatures
from criterions.mean_squared_error import MeanSquaredError, SquaredError
from models.linear_regressor import LinearRegressor
from dataset.points_dataset import PointsDataset, Point
import matplotlib.pyplot as plt


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
        if len(data) < self.sample_size:
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
    ransac = RANSAC(
        model=LinearRegressor(num_features=1),
        criterion=SquaredError(),
        loss=MeanSquaredError(),
        n_iterations=100000,
        threshold=0.05,
        sample_size=10,
        datapoints_required=10
    )

    X = np.array([-0.848,-0.800,-0.704,-0.632,-0.488,-0.472,-0.368,-0.336,-0.280,-0.200,-0.00800,-0.0840,0.0240,0.100,0.124,0.148,0.232,0.236,0.324,0.356,0.368,0.440,0.512,0.548,0.660,0.640,0.712,0.752,0.776,0.880,0.920,0.944,-0.108,-0.168,-0.720,-0.784,-0.224,-0.604,-0.740,-0.0440,0.388,-0.0200,0.752,0.416,-0.0800,-0.348,0.988,0.776,0.680,0.880,-0.816,-0.424,-0.932,0.272,-0.556,-0.568,-0.600,-0.716,-0.796,-0.880,-0.972,-0.916,0.816,0.892,0.956,0.980,0.988,0.992,0.00400]).reshape(-1,1)
    y = np.array([-0.917,-0.833,-0.801,-0.665,-0.605,-0.545,-0.509,-0.433,-0.397,-0.281,-0.205,-0.169,-0.0531,-0.0651,0.0349,0.0829,0.0589,0.175,0.179,0.191,0.259,0.287,0.359,0.395,0.483,0.539,0.543,0.603,0.667,0.679,0.751,0.803,-0.265,-0.341,0.111,-0.113,0.547,0.791,0.551,0.347,0.975,0.943,-0.249,-0.769,-0.625,-0.861,-0.749,-0.945,-0.493,0.163,-0.469,0.0669,0.891,0.623,-0.609,-0.677,-0.721,-0.745,-0.885,-0.897,-0.969,-0.949,0.707,0.783,0.859,0.979,0.811,0.891,-0.137]).reshape(-1,1)

    data = np.hstack((X, y))
    print('Data shape:', data.shape)
    dataset = PointsDataset([Point(x, y) for x, y in data])
    print('Dataset length:', len(dataset))

    ransac.fit(dataset, verbose=True)
    ransac.visualize(dataset)