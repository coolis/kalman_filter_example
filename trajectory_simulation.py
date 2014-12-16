import numpy as np
import utils
import kalman_filter
import matplotlib.pyplot as plt
import os

baseDir = os.path.dirname(__file__)

def trajectory_simulation(x, y, vx, vy, ax, ay, T):
    x_t = []
    for t in range(T):
        x_t.append([x+vx*t+0.5*ax*t**2, vx+ax*t, y+vy*t+0.5*ay*t**2, vy+ay*t])
    return x_t

def trajectory_simulation_noise(x, y, vx, vy, ax, ay, T):
    x_t_noise = []
    for t in range(T):
        x_t_noise.append([x+vx*t+0.5*ax*t**2, vx+ax*t, y+vy*t+0.5*ay*t**2, vy+ay*t] + np.random.normal(0, 1000**0.5, 4))
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

# 2. Plot the trajectories

# with noise
utils.line(x_f[:,0], x_f[:, 2], "x", "y", "ground-truth trajectory without noise")

# without noise
utils.line(x_f_n[:,0], x_f_n[:, 2], "x", "y", "ground-truth trajectory with noise")
utils.line(np.arange(500), x_f_n[:,1], "time", "velocity", "velocity of x")
utils.line(np.arange(500), x_f_n[:,3], "time", "velocity", "velocity of y")
