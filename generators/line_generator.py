from .generator import Generator
from dataset.points_dataset import PointsDataset, Point
import numpy as np
import matplotlib.pyplot as plt


class LineGenerator(Generator):
    '''2D Line Generator class for generating points on a random line with configurable noise'''
    def __init__(self, 
                 num_samples, 
                 noise_level, 
                 x_range=(-10, 10), 
                 slope_range=(-5, 5), 
                 intercept_range=(-10, 10), 
                 jitter=0.05, 
                 salt_pepper_ratio=0.5):
        '''
        Args:
            num_samples (int): Number of samples to generate.
            noise_level (float): Proportion of points to be outliers.
            x_range (tuple): Range for x values.
            slope_range (tuple): Range for the slope of the line.
            intercept_range (tuple): Range for the y-intercept of the line.
            jitter (float): Standard deviation of Gaussian noise added to inliers.
            salt_pepper_ratio (float): Ratio of salt and pepper noise applied to outliers.
        '''
        self.num_samples = num_samples
        self.noise_level = noise_level
        self.x_range = x_range
        self.slope_range = slope_range
        self.intercept_range = intercept_range
        self.jitter = jitter
        self.salt_pepper_ratio = salt_pepper_ratio

        # populated by generate()
        self.slope = None
        self.intercept = None
        self.data = None
        self.inlier_mask = None

    def generate(self):
        self.slope = np.random.uniform(*self.slope_range)
        self.intercept = np.random.uniform(*self.intercept_range)

        # 'perfect' inliers
        x_values = np.random.uniform(*self.x_range, self.num_samples)
        y_values = self.slope * x_values + self.intercept

        # jittered inliers
        y_jittered = y_values + np.random.normal(0, self.jitter, self.num_samples)
        self.data = np.column_stack((x_values, y_jittered))
        self.inlier_mask = np.ones(self.num_samples, dtype=bool)

        self._apply_salt_pepper_noise()

        return PointsDataset([Point(x, y) for x, y in self.data])

    def _apply_salt_pepper_noise(self):
        num_outliers = int(self.num_samples * self.noise_level)
        if num_outliers == 0:
            return
        
        # randomly selected indices to corrupt
        outlier_indices = np.random.choice(self.num_samples, num_outliers, replace=False)
        self.inlier_mask[outlier_indices] = False

        num_salt = int(num_outliers * self.salt_pepper_ratio)
        salt_indices = outlier_indices[:num_salt]
        pepper_indices = outlier_indices[num_salt:]

        y_min, y_max = self.data[:, 1].min(), self.data[:, 1].max()
        data_range = y_max - y_min
        salt_value = y_max + data_range * np.random.uniform(0, 1, len(salt_indices))
        pepper_value = y_min - data_range * np.random.uniform(0, 1, len(pepper_indices))

        self.data[salt_indices, 1] = salt_value
        self.data[pepper_indices, 1] = pepper_value

    def visualize(self):
        if self.data is None:
            raise RuntimeError('Data has not been generated yet. Call generate() before visualize().')
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data[:, 0], self.data[:, 1], label='Data Points', s=20, color='blue')
        plt.title('Generated Line with Salt-Pepper Noise')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()