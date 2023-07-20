from dataclasses import dataclass

import numpy as np


class Rotation:
    def __init__(self):
        pass

    @dataclas
    def from_euler(self, seq='zyx', angles, degrees='False'):
        pass

    def from_quat(self, quat):
        pass

    def from_rotmat(self, mat):
        pass

    def from_rotvec(self, vec, degrees='False'):
        pass
    
    def as_euler(self, seq='zyx', degrees='False'):
        pass
    
    def as_quat(self):
        pass

    def as_rotmat(self):
        pass

    def as_rotvec(self, degrees='False'):
        pass

    def apply(self, vec):
        pass

    def identity(self):
        pass

    def random(self, n: int):
        pass
