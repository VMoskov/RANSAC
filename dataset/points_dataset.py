from dataclasses import dataclass
import numpy as np


@dataclass
class Point:
    x: float
    y: float


class PointsDataset:
    def __init__(self, points):
        self.points = points
    
    @property
    def shape(self):
        return (len(self.points), 2)

    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, key):
        if isinstance(key, int):  # single index
            return self.points[key]

        if isinstance(key, slice):  # slice indexing
            return PointsDataset(self.points[key])

        if isinstance(key, (list, np.ndarray)):  # mask
            if hasattr(key, 'dtype') and key.dtype == bool:
                filtered_points = [p for p, m in zip(self.points, key) if m]
                return PointsDataset(filtered_points)
            
            filtered_points = [self.points[i] for i in key]
            return PointsDataset(filtered_points)
            
        raise TypeError(f"PointsDataset indices must be integers, slices, or a boolean mask, not {type(key)}")

    
    def to_numpy(self):
        return np.array([[point.x, point.y] for point in self.points])