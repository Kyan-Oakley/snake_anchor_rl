from scipy.spatial import ConvexHull

class ConvexHullEval(ConvexHull):
    def __init__(self, points):
        super().__init__(points)

    def epsilon_metric(self):
        return float(min(-eqn[-1] for eqn in self.equations))