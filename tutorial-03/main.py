from dataclasses import dataclass

import numpy as np


class Rotation:
    def __init__(self, quat):
        self._quat = quat

    @classmethod
    def from_euler(cls, seq, angles):
        for angle in angles:
            quat = Quaternion.from_rotvec(mu, n)
        pass

    @classmethod
    def from_quat(cls, quat):
        return cls(quat)

    def as_quat(self):
        pass

    def as_matrix(self):
        pass

    def apply(self, vector):
        pass

    def __mul__(self, other):
        return Rotation.from_quat(self._quat * other._quat)


@dataclass
class Euler(Rotation):
    a1: float = 0.0
    a2: float = 0.0
    a3: float = 0.0
    order: str = "ZYX"


@dataclass
class Complex:
    real: float
    imag: int = 0

    def __mul__(self, other):
        return Complex(self.real * other.real, other.imag - self.imag)


class Quaternion(Rotation):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        p0, p1, p2, p3 = self.value.ravel()
        return f"{p0:5.2f} {p1:+5.2f} i {p2:+5.2f} j {p3:+5.2f} k"

    @property
    def vector(self):
        return self.value[1:]

    @property
    def scalar(self):
        return self.value[:1]

    def __add__(self, other):
        return Quaternion(self.value + other.value)

    def __sub__(self, other):
        return Quaternion(self.value - other.value)

    def __mul__(self, other):
        p0, p = self.scalar, self.vector
        q0, q = other.scalar, other.vector

        return Quaternion(
            np.vstack(
                (
                    p0 * q0 - p.T @ q,
                    p0 * q + q0 * p + np.cross(p, q, axis=0),
                )
            )
        )

    def norm(self):
        return np.sum(self.value**2)

    def inverse(self):
        return Quaternion(1 / self.norm() * np.vstack((self.scalar, -self.vector)))


q1 = Quaternion(np.vstack((1.0, 0.0, 2.0, 1.0)))
q2 = Quaternion(np.vstack((2.0, 1.0, 1.0, -1.0)))
print((q1 * q2).inverse())
print(q2.inverse() * q1.inverse())

rot = Rotation.from_euler("ZYX", [10, 20, 30])
rot = rot * rot * rot
rot.apply(vector)

R = rot.as_matrix()
print(R @ vector)
print(rot.apply(vector))

q = rot.as_quat()

# rot = Euler(*np.deg2rad([10, 20, 30]), order="ZYX")
# rot.to_quat()
# rot.to_dcm()
