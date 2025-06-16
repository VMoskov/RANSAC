import numpy as np


class SquaredError:
    '''Squared Error Criterion'''
    def __call__(self, y_true, y_pred):
        '''Returns element-wise squared error between true and predicted values'''
        if y_true.shape != y_pred.shape:
            raise ValueError('Shapes of y_true and y_pred must match.')
        return (y_true - y_pred) ** 2
    

class MeanSquaredError:
    '''Mean Squared Error Loss Function'''
    def __call__(self, y_true, y_pred):
        '''Returns mean squared error between true and predicted values'''
        if y_true.shape != y_pred.shape:
            raise ValueError('Shapes of y_true and y_pred must match.')
        return np.mean((y_true - y_pred) ** 2)