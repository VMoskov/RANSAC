import random
import numpy as np
import warnings
from sklearn.preprocessing import PolynomialFeatures


class RANSAC:
    def __init__(self, model, criterion, loss, n_iterations=1000, threshold=1.0, sample_size=20):
        self.model = model
        self.criterion = criterion  # criterion function to evaluate model predictions
        self.loss = loss   # loss function to evaluate model performance

        self.n_iterations = n_iterations
        self.threshold = threshold
        self.sample_size = sample_size

        self.best_model = model


    def fit(self, data, verbose=False):
        best_inliers_mask = np.zeros(data.shape[0], dtype=bool)
        best_loss = float('inf')
        
        for _ in range(self.n_iterations):
            sample = self._random_sample(data)

            X, y = self._transform_data(sample)

            model_candidate = self.model.fit(X, y)
            inliers_mask = self._get_inliers_mask(X, y, model_candidate)

            if inliers_mask.sum() > best_inliers_mask.sum():
                best_inliers_mask = inliers_mask
                # we don't calculate loss here, more inliers is always better

        if best_inliers_mask.any():
            final_inliers = data[best_inliers_mask]

            X_final, y_final = self._transform_data(final_inliers)

            self.best_model = self.model.fit(X_final, y_final)
            self.best_inliers = final_inliers

            if verbose: self._diagnostics(data)

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

    def _random_sample(self, data):
        if len(data) < self.sample_size:
            raise ValueError('Not enough data points to sample from.')
        sample_indices = random.sample(data.shape[0], self.sample_size)
        return data[sample_indices]

    def _get_inliers_mask(self, X, y, model):
        preds = model.predict(X)
        losses = self.criterion(y, preds)
        return losses < self.threshold

    def _transform_data(self, data):
        if hasattr(data, 'to_numpy'):
            X, y = data.to_numpy()
            return X, y
        
        if isinstance(data, np.ndarray):
            # assume 2D case (x, y)
            X = data[:, 0]
            y = data[:, 1]
            X = PolynomialFeatures(degree=1, include_bias=True).fit_transform(X.reshape(-1, 1))
            return X, y
        
        if isinstance(data, list):
            # assume list of tuples (x, y)
            X = np.array([[1, x] for x, _ in data])
            y = np.array([y for _, y in data])
            return X, y
        
        raise ValueError('Unsupported data format. Provide a numpy array, list of tuples, or a dataset with to_numpy method.')
    
    def _diagnostics(self, data):
        inliers_pred = self.predict(self.best_inliers)
        inliers_loss = self.loss(self.best_inliers, inliers_pred)

        outliers_mask = ~self._get_inliers_mask(data, self.best_model)
        outliers_pred = self.predict(data[outliers_mask])
        outliers_loss = self.loss(data[outliers_mask], outliers_pred)

        data_loss = self.loss(data, self.predict(data))
        print(f'Inliers Loss: {inliers_loss}, Outliers Loss: {outliers_loss}, Total Loss: {data_loss}')
