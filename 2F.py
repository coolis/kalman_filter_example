import numpy as np
import utils
import kalman_filter
import matplotlib.pyplot as plt
import os
import random

baseDir = os.path.dirname(__file__)

def trajectory_simulation(x, y, vx, vy, ax, ay, T):
    x_t = []
    for t in range(T):
        x_t.append([x+vx*t+0.5*ax*t**2, vx+ax*t, y+vy*t+0.5*ay*t**2, vy+ay*t])
    return x_t

def trajectory_simulation_noise(x, y, vx, vy, ax, ay, T):
    x_t_noise = []
    for t in range(T):
        x_t_noise.append([x+vx*t+0.5*ax*t**2, vx+ax*t, y+vy*t+0.5*ay*t**2, vy+ay*t] + np.random.normal(0, 1000, 4))
    return x_t_noise

# ===========================================
# Task 1 Simulation

# 1. Generate ground-truth trajectories
x1 = trajectory_simulation(0, 0, 300, 0, 0, 0, 200)
x2 = trajectory_simulation(x1[len(x1)-1][0], x1[len(x1)-1][2], 300, 0, 2, 2, 100)
x3 = trajectory_simulation(x2[len(x2)-1][0], x2[len(x2)-1][2], 0, 300, 0, 0, 200)

x1_n = trajectory_simulation_noise(0, 0, 300, 0, 0, 0, 200)
x2_n = trajectory_simulation_noise(x1_n[len(x1_n)-1][0], x1_n[len(x1_n)-1][2], 300, 0, 2, 2, 100)
x3_n = trajectory_simulation_noise(x2_n[len(x2_n)-1][0], x2_n[len(x2_n)-1][2], 0, 300, 0, 0, 200)

x_f = x1+x2+x3
x_f = np.asmatrix(x_f)

x_f_n = x1_n+x2_n+x3_n
x_f_n = np.asmatrix(x_f_n)

# F. reduce error rate
#initial values
A = [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
B = [[0.5, 0], [1, 0], [0, 0.5], [0, 1]]
H = [[1, 0, 0, 0], [0, 0, 1, 0]]
x = [0, 300, 0, 0]
P = [[0, 0, 0, 0], [0, 10000, 0, 0], [0, 0, 0, 0], [0, 0, 0, 10000]]
Q = np.asmatrix([[0.3**2, 0], [0, 0.3**2]])
R = np.asmatrix([[1000**2, 0], [0, 1000**2]])
kfl = kalman_filter.KalmanFilterLinear(np.asmatrix(A), np.asmatrix(B), np.asmatrix(H), np.asmatrix(x).T, np.asmatrix(P), Q, R)

for t in range(500):
    kfl.Step(np.asmatrix(x_f_n[t, [0,2]]).T)
    if (t == 0):
        estimate = kfl.GetCurrentState()
    else:
        estimate = np.concatenate((estimate, kfl.GetCurrentState()), axis=1)

kfl_random = kalman_filter.KalmanFilterLinearRandom(np.asmatrix(A), np.asmatrix(B), np.asmatrix(H), np.asmatrix(x).T, np.asmatrix(P), Q, R)

for t in range(500):
    r = random.random()
    if (r < 0.5):
        kfl_random.Step(np.asmatrix(x_f_n[t, [0,2]]).T)
    else:
        kfl_random.Step(np.asmatrix(np.zeros((0,0))))
    if (t == 0):
        estimate_random = kfl_random.GetCurrentState()
    else:
        estimate_random = np.concatenate((estimate_random, kfl_random.GetCurrentState()), axis=1)

fig = plt.figure()
plt.title("2F.Plot of estimated trajectory, raw measurement, and true trajectory")
plt.ylabel("y")
plt.xlabel("x")
plt.plot(estimate[0, :].T, estimate[2, :].T)
plt.plot(estimate_random[0, :].T, estimate_random[2, :].T)
plt.plot(x_f[:, 0], x_f[:, 2])
plt.plot(x_f_n[:, 0], x_f_n[:, 2])
fig.savefig(os.path.join(baseDir, 'Figures/2F.Estimated measurement and true trajectory.png'))
