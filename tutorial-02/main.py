import fym
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, tan
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation


class Quadrotor(fym.BaseEnv):
    def __init__(self):
        super().__init__(max_t=10)
        self.pos = fym.BaseSystem(shape=(3, 1))
        self.vel = fym.BaseSystem(shape=(3, 1))
        self.angles = fym.BaseSystem(shape=(3, 1))
        self.omega = fym.BaseSystem(shape=(3, 1))

    def set_dot(self, t):
        pos, vel, angles, omega = self.observe_list()

        m = 0.96
        g = 9.81
        e3 = np.vstack((0, 0, 1))
        J = np.vstack((0.039, 0.034, 0.071))

        b = 0.16
        sigma = 2.1e-9 / 8.67e-8
        B = np.array(
            [
                [1, 1, 1, 1],
                [0, -b, 0, b],
                [b, 0, -b, 0],
                [-sigma, sigma, -sigma, sigma],
            ]
        )

        u = np.ones((4, 1)) * m * g / 4 * 0.98

        fM = B @ u
        f, M = fM[:1], fM[1:]

        angles = angles.ravel()

        R = Rotation.from_euler("ZYX", angles[::-1]).as_matrix()

        phi, theta, psi = angles
        H = np.array(
            (
                [1, sin(phi) * tan(theta), cos(phi) * tan(theta)],
                [0, cos(phi), -sin(phi)],
                [0, sin(phi) / cos(theta), cos(phi) / cos(theta)],
            )
        )

        self.pos.dot = vel
        self.vel.dot = (1 / m) * (m * g * e3 - R @ e3 * f)
        self.angles.dot = H @ omega
        self.omega.dot = (1 / J) * (-np.cross(omega, J * omega, axis=0) + M)

        return {"t": t, "pos": pos, "u": u}


env = Quadrotor()
env.logger = fym.Logger(mode="stop")

while True:
    env.render()

    _, done = env.update()

    if done:
        break

plt.plot(env.logger.buffer["t"], env.logger.buffer["u"][:, 0], label="z")
plt.show()
