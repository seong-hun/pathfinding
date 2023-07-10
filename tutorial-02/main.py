import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, tan
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation


def nonlinear_system(t, x):
    pos = x[:3]
    vel = x[3:6]
    angles = x[6:9]
    omega = x[9:]

    m = 0.96
    g = 9.81
    e3 = np.vstack((0, 0, 1))
    J = np.array([0.039, 0.034, 0.071])

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

    u = np.ones(4) * m * g / 4 * 0.98

    fM = B @ u
    f, M = fM[:1], fM[1:]

    R = Rotation.from_euler("ZYX", angles[::-1]).as_matrix()

    phi, theta, psi = angles
    H = np.array(
        (
            [1, sin(phi) * tan(theta), cos(phi) * tan(theta)],
            [0, cos(phi), -sin(phi)],
            [0, sin(phi) / cos(theta), cos(phi) / cos(theta)],
        )
    )

    pos_dot = vel
    vel_dot = (1 / m) * (m * g * e3 - R @ e3 * f).ravel()
    angles_dot = H @ omega
    omega_dot = (1 / J) * (-np.cross(omega, J * omega) + M)

    return np.hstack((pos_dot, vel_dot, angles_dot, omega_dot))


x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
sol = solve_ivp(nonlinear_system, t_span=(0, 10), t_eval=np.linspace(0, 10, 100), y0=x0)

pos = sol.y[:3]
vel = sol.y[3:6]
angles = sol.y[6:9]
omega = sol.y[9:]

plt.figure()
plt.plot(sol.t, pos[2], label="z")

plt.figure()
plt.plot(sol.t, omega[0], label="p")

plt.show()
