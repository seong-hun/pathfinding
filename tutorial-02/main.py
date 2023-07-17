from dataclasses import dataclass, field

import fym
import matplotlib.pyplot as plt
import numpy as np
from fym.utils.rot import quat2dcm
from numpy import cos, sin, tan
from scipy.spatial.transform import Rotation


class Controller(fym.BaseSystem):
    def __init__(self):
        super().__init__([])

    def set_dot(self):
        raise NotImplementedError


class PD(Controller):
    def set_dot(self):
        pass

    def get(self):
        plant = self.base_env.plant

        pos = plant.pos.state
        vel = plant.vel.state

        z = pos[2]
        vz = vel[2]

        zd, vzd = self.base_env.ref.get()

        Kp = 10
        Kd = 5

        fc0 = plant.m * plant.g
        fc = fc0 + Kp * (z - zd) + Kd * (vz - vzd)
        u = np.linalg.pinv(plant.B) @ np.vstack((fc, 0, 0, 0))

        return u


class PID(Controller):
    def __init__(self):
        fym.BaseSystem.__init__(self, shape=(1,))

    def set_dot(self):
        pos = self.base_env.plant.pos.state
        z = pos[2]
        zd, _ = self.base_env.ref.get()
        self.dot = z - zd

    def get(self):
        plant = self.base_env.plant
        pos = plant.pos.state
        vel = plant.vel.state
        ei = self.state  # zi - zid

        z = pos[2]
        vz = vel[2]

        zd, vzd = self.base_env.ref.get()

        Kp = 10
        Kd = 5
        Ki = 3

        fc0 = plant.m * plant.g
        fc = fc0 + Kp * (z - zd) + Kd * (vz - vzd) + Ki * ei
        u = np.linalg.pinv(plant.B) @ np.vstack((fc, 0, 0, 0))

        return u


class RotSystem(fym.BaseSystem):
    def get_R(self):
        raise NotImplementedError

    def set_dot(self):
        raise NotImplementedError


class Euler(RotSystem):
    def __init__(self, initial_state=np.zeros((3, 1))):
        super().__init__(initial_state)

    def get_angles(self):
        return self.state.ravel()

    def get_R(self):
        angles = self.get_angles()
        return Rotation.from_euler("ZYX", angles[::-1]).as_matrix()

    def set_dot(self):
        phi, theta, _ = self.get_angles()

        H = np.array(
            (
                [1, sin(phi) * tan(theta), cos(phi) * tan(theta)],
                [0, cos(phi), -sin(phi)],
                [0, sin(phi) / cos(theta), cos(phi) / cos(theta)],
            )
        )

        omega = self.base_env.plant.omega.state
        self.dot = H @ omega


class Quaternion(RotSystem):
    def __init__(self, initial_state=np.vstack((1.0, 0, 0, 0))):
        super().__init__(initial_state)

    def set_dot(self):
        quat = self.state
        p, q, r = self.base_env.plant.omega.state.ravel()
        dquat = (
            0.5
            * np.array(
                (
                    [0, -p, -q, -r],
                    [p, 0, r, -q],
                    [q, -r, 0, p],
                    [r, q, -p, 0],
                )
            )
            @ quat
        )
        eps = 1 - (quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2)
        self.dot = dquat + eps * quat

    def get_R(self):
        quat = self.state
        return quat2dcm(quat).T


class Plant(fym.BaseEnv):
    def set_dot(self):
        raise NotImplementedError


class Quadrotor(Plant):
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

    def __init__(self, rot: type = Euler, **kwargs):
        super().__init__(**kwargs)
        self.pos = fym.BaseSystem(shape=(3, 1))
        self.vel = fym.BaseSystem(shape=(3, 1))
        self.rot = rot()
        self.omega = fym.BaseSystem(shape=(3, 1))

    def set_dot(self):
        _, vel, _, omega = self.observe_list()
        u = self.base_env.controller.get()

        e3 = np.vstack((0, 0, 1))
        J = np.vstack([0.039, 0.034, 0.071])

        fM = self.B @ u
        f, M = fM[:1], fM[1:]

        R = self.rot.get_R()

        self.pos.dot = vel
        self.vel.dot = (1 / self.m) * (self.m * self.g * e3 - R @ e3 * f)
        self.rot.set_dot()
        self.omega.dot = (1 / J) * (-np.cross(omega, J * omega, axis=0) + M)

        return self.observe_dict()


class Reference(fym.BaseSystem):
    def __init__(self):
        super().__init__([])

    def get(self):
        return 0, 0


class Step(Reference):
    def __init__(self, height, t=0):
        super().__init__()
        self.height = height
        self.t = t

    def get(self):
        if self.base_env.t > self.t:
            return self.height, 0
        else:
            return 0, 0


class Env(fym.BaseEnv):
    def __init__(self, plant: Plant, controller: Controller, ref: Reference, **kwargs):
        super().__init__(**kwargs)
        self.plant = plant
        self.controller = controller
        self.ref = ref

    def set_dot(self, t):
        cinfo = self.controller.set_dot() or {}
        pinfo = self.plant.set_dot() or {}
        return {"t": t, **pinfo, **cinfo}

    def run(self):
        self.logger = fym.Logger(mode="stop")

        while True:
            self.render()

            _, done = self.update()

            if done:
                break

        return self.logger.buffer


def run(exps):
    for exp in exps:
        exp.data = exp.env.run()


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
    style: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)


def script1():
    exps = [
        Exp(
            env=Env(
                plant=Quadrotor(),
                controller=PID(),
                ref=Step(-1),
                max_t=10,
            ),
            style={"label": "PID", "c": "k"},
        ),
        Exp(
            env=Env(
                plant=Quadrotor(),
                controller=PD(),
                ref=Step(-1),
                max_t=10,
            ),
            style={"label": "PD", "c": "r", "ls": "--"},
        ),
    ]

    run(exps)

    plot_exps(exps)


def script2():
    exps = [
        Exp(
            env=Env(
                plant=Quadrotor(rot=Euler),
                controller=PID(),
                ref=Step(-1),
                max_t=10,
            ),
            style={"label": "Euler", "c": "b"},
        ),
        Exp(
            env=Env(
                plant=Quadrotor(rot=Quaternion),
                controller=PID(),
                ref=Step(-1),
                max_t=10,
            ),
            style={"label": "Quaternion", "c": "r", "ls": "--"},
        ),
    ]

    run(exps)

    plot_exps(exps)


def script3():
    PID_style = {"c": "b"}
    PD_style = {"c": "r", "ls": "--"}

    exps = [
        Exp(
            env=Env(
                plant=Quadrotor(),
                controller=PID(),
                ref=Step(-1),
                max_t=10,
            ),
            style={"label": "PID", **PID_style},
        ),
        Exp(
            env=Env(
                plant=Quadrotor(),
                controller=PD(),
                ref=Step(-1),
                max_t=10,
            ),
            style={"label": "PD", **PD_style},
        ),
        Exp(
            env=Env(
                plant=Quadrotor(),
                controller=PID(),
                ref=Step(-2),
                max_t=10,
            ),
            style=PID_style,
        ),
        Exp(
            env=Env(
                plant=Quadrotor(),
                controller=PD(),
                ref=Step(-2),
                max_t=10,
            ),
            style=PD_style,
        ),
    ]

    run(exps)

    plot_exps(exps)


def script4():
    exps = [
        Exp(
            env=Env(
                plant=Quadrotor(),
                controller=PD(),
                ref=Step(-2, t=3),
                max_t=10,
                solver="odeint",
            ),
            style={"label": "odeint", "c": "k", "ls": "-"},
        ),
        Exp(
            env=Env(
                plant=Quadrotor(),
                controller=PD(),
                ref=Step(-2, t=3),
                max_t=10,
                solver="rk4",
            ),
            style={"label": "rk4", "c": "r", "ls": "--"},
        ),
        Exp(
            env=Env(
                plant=Quadrotor(),
                controller=PD(),
                ref=Step(-2, t=3),
                dt=10,
                ode_step_len=1000,
                max_t=10,
                solver="odeint",
            ),
            style={"label": "odeint (one-step)", "c": "b", "ls": "-."},
        ),
        Exp(
            env=Env(
                plant=Quadrotor(),
                controller=PD(),
                ref=Step(-2, t=3),
                dt=10,
                ode_step_len=1000,
                max_t=10,
                solver="rk4",
            ),
            style={"label": "rk4 (one-step)", "c": "g", "ls": ":"},
        ),
    ]

    run(exps)

    plot_exps(exps)


if __name__ == "__main__":
    # script1()
    # script2()
    # script3()
    script4()
