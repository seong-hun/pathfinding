from dataclasses import dataclass

import numpy as np


class Rotation:
    def __init__(self):
        pass

    @dataclass
    def from_euler(self):
        pass

    def from_quat(self):
        pass

    def from_rotmat(self):
        pass

    def from_rotvec(self):
        pass
    
    def as_euler(self):
        pass
    
    def as_quat(self):
        pass

    def as_rotmat(self):
        pass

    def as_rotvec(self):
        pass

    def apply(self):
        pass

    def identity(self):
        pass

    def random(self, n: int):
        pass
