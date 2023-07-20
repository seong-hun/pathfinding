import numpy as np


class Quaternion:
    def __init__(self, p0: float, p1: float, p2: float, p3: float):
        self.p0 = p0
        self.pvec = np.vstack((p1, p2, p3))
