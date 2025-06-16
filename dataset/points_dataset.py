from dataclasses import dataclass
import numpy as np


@dataclass
class Point:
    x: float
    y: float


class PointsDataset:
    def __init__(self, points):
        self.points = points

    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.points):
            raise IndexError('Index out of bounds')
        return self.points[idx]
    
    def to_numpy(self):
        return np.array([(point.x, point.y) for point in self.points])