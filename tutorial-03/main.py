"""
Test Scripts
"""

from element import Quaternion
from transform import Rotation


def test_create_quaternion():
    q1 = Quaternion(1, 0, 0, 0)
    q2 = Quaternion(0, 1, 0, 0)


def test_repr_quaternion():
    pass


def test_multiply_quaternion():
    pass


if __name__ == "__main__":
    test_create_quaternion()
    test_repr_quaternion()
    test_multiply_quaternion()
