import random
import numpy as np
import warnings


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
        data = self._transform_data(data)  # ensure data is in numpy array format
        best_inliers_mask = np.zeros(data.shape[0], dtype=bool)
        best_loss = float('inf')
        
        for _ in range(self.n_iterations):
            sample = self._random_sample(data)
            model_candidate = self.model.fit(sample)
            inliers_mask = self._get_inliers_mask(data, model_candidate)

            if inliers_mask.sum() > best_inliers_mask.sum():
                best_inliers_mask = inliers_mask
                # we don't calculate loss here, more inliers is always better

        if best_inliers_mask.any():
            final_inliers = data[best_inliers_mask]
            self.best_model = self.model.fit(final_inliers)
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
        
        data = self._transform_data(data)
        return self.best_model.predict(data)

    def _random_sample(self, data):
        if len(data) < self.sample_size:
            raise ValueError('Not enough data points to sample from.')
        sample_indices = random.sample(data.shape[0], self.sample_size)
        return data[sample_indices]

    def _get_inliers_mask(self, data, model):
        preds = model.predict(data)
        losses = self.criterion(data, preds)
        return losses < self.threshold

    def _transform_data(self, data):
        if hasattr(data, 'to_numpy'):
            return data.to_numpy()
        
        if isinstance(data, np.ndarray):
            return data
        
        if isinstance(data, list):
            if len(data) == 0:
                return np.array([])
            
            if hasattr(data[0], 'x') and hasattr(data[0], 'y'):  # Check if elements are objects with x,y and convert, otherwise just convert the list
                return np.array([[p.x, p.y] for p in data])
            else:
                return np.array(data)
        
        raise TypeError('Unsupported data type. Data must be a NumPy array, a list, or have a .to_numpy() method.')
    
    def _diagnostics(self, data):
        inliers_pred = self.predict(self.best_inliers)
        inliers_loss = self.loss(self.best_inliers, inliers_pred)

        outliers_mask = ~self._get_inliers_mask(data, self.best_model)
        outliers_pred = self.predict(data[outliers_mask])
        outliers_loss = self.loss(data[outliers_mask], outliers_pred)

        data_loss = self.loss(data, self.predict(data))
        print(f'Inliers Loss: {inliers_loss}, Outliers Loss: {outliers_loss}, Total Loss: {data_loss}')
