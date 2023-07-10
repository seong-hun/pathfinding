from dataclasses import dataclass

import fym
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, tan
from scipy.spatial.transform import Rotation


class Controller:
    def set_dot(self, plant):
        raise NotImplementedError


class PD(Controller):
    def set_dot(self, plant):
        pos = plant.pos.state
        vel = plant.vel.state

        z = pos[2]
        vz = vel[2]

        zd = -1
        vzd = 0

        Kp = 10
        Kd = 5

        fc0 = plant.m * plant.g
        fc = fc0 + Kp * (z - zd) + Kd * (vz - vzd)
        u = np.linalg.pinv(plant.B) @ np.vstack((fc, 0, 0, 0))

        return u


class PID(Controller, fym.BaseSystem):
    def set_dot(self, plant):
        pos = plant.pos.state
        vel = plant.vel.state

        z = pos[2]
        vz = vel[2]

        zd = -1
        vzd = 0

        ei = self.state  # zi - zid

        Kp = 10
        Kd = 5
        Ki = 3

        fc0 = plant.m * plant.g
        fc = fc0 + Kp * (z - zd) + Kd * (vz - vzd) + Ki * ei
        u = np.linalg.pinv(plant.B) @ np.vstack((fc, 0, 0, 0))

        self.dot = z - zd

        return u


class Quadrotor(fym.BaseEnv):
    m = 0.96
    g = 9.81

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pos = fym.BaseSystem(shape=(3, 1))
        self.vel = fym.BaseSystem(shape=(3, 1))
        self.angles = fym.BaseSystem(shape=(3, 1))
        self.omega = fym.BaseSystem(shape=(3, 1))

    def set_dot(self, u):
        pos, vel, angles, omega = self.observe_list()

        e3 = np.vstack((0, 0, 1))
        J = np.vstack([0.039, 0.034, 0.071])

        fM = self.B @ u
        f, M = fM[:1], fM[1:]

        angles = angles.ravel()
        phi, theta, psi = angles
        R = Rotation.from_euler("ZYX", angles[::-1]).as_matrix()

        H = np.array(
            (
                [1, sin(phi) * tan(theta), cos(phi) * tan(theta)],
                [0, cos(phi), -sin(phi)],
                [0, sin(phi) / cos(theta), cos(phi) / cos(theta)],
            )
        )

        self.pos.dot = vel
        self.vel.dot = (1 / self.m) * (self.m * self.g * e3 - R @ e3 * f)
        self.angles.dot = H @ omega
        self.omega.dot = (1 / J) * (-np.cross(omega, J * omega, axis=0) + M)

        return self.observe_dict()


class Env(fym.BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plant = Quadrotor()
        self.controller = Controller()

    def set_dot(self, t):
        u = self.controller.set_dot(self.plant)
        pinfo = self.plant.set_dot(u)
        return {"t": t, **pinfo}


def run(env, controller_cls):
    env = Env(dt=0.01, max_t=10)
    env.controller = controller_cls()
    env.logger = fym.Logger(mode="stop")

    while True:
        env.render()

        _, done = env.update()

        if done:
            break

    return env.logger.buffer


def plot_exps(exps):
    plt.figure()

    for exp in exps:
        t = exp.data["t"]
        pos = exp.data["pos"]
        plt.plot(t, -pos[:, 2], **exp.style)

    plt.legend()

    plt.figure()

    for exp in exps:
        t = exp.data["t"]
        omega = exp.data["omega"]
        plt.plot(t, omega[:, 0], **exp.style)

    plt.legend()

    plt.show()


@dataclass
class Exp:
    env: Env
    controller_cls: ...
    data: ... = None
    style: ... = None


def main():
    env = Env()
    exps = [
        Exp(env, PID, style={"label": "PID", "c": "k"}),
        Exp(env, PD, style={"label": "PD", "c": "r", "ls": "--"}),
    ]

    for exp in exps:
        exp.data = run(exp.env, exp.controller_cls)

    plot_exps(exps)


if __name__ == "__main__":
    main()
