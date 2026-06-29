from scipy.spatial import ConvexHull

class ConvexHullEval(ConvexHull):
    def __init__(self, points):
        super().__init__(points)

    def max_inscribed_circle(self):
        for eqn in self.equations:
            print(eqn)