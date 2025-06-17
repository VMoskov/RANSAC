class PointsDataset:
    def __init__(self, points):
        self.points = points
    
    @property
    def shape(self):
        return self.points.shape

    def __len__(self):
        return self.points.shape[0]
    
    def __getitem__(self, key):
        return self.points[key]

    def to_numpy(self):
        return self.points