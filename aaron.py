import numpy as np

import matplotlib.pyplot as plt

ps = 1E-15
c = 3E8
delta_t = 10 * ps  # time step
delta_x = c * delta_t * 3
num_points = 100


def z_line(x):
    return 50

def stimulus(t):
    tau = 30 * delta_t
    t0 = 3 * tau
    return np.exp(-((t - t0) / tau)**2)


# sequence a, b, c, d
# c, d are in the future
# a, b are in the past
v_a = np.zeros(num_points)
i_b = np.zeros(num_points)
v_c = np.zeros(num_points)
i_d = np.zeros(num_points)



num_cycles = 10000
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
x = np.array(range(0, num_points)) * delta_x
line, = ax.plot(x, v_c)

# Update function
def update_plot(new_x, new_y):
    ax.cla() # Clear the current axes
    ax.plot(new_x, new_y) # Redraw with updated data
    plt.draw()

for n in range(0, num_cycles):
    time = n * delta_t
    z0 = 50
    vp = c
    L = z0 / vp
    C = 1 / (z0 * vp)
    v_c[0] = stimulus(time)

    # step 1, time step
    v_c[1:] = v_a[1:] - delta_t / (C * delta_x) * (i_b[1:] - i_b[0:-1])
    i_d[0:-1] = i_b[0:-1] - delta_t / (L * delta_x) * (v_c[1:] - v_c[0:-1])

    # step 2, space step
    i_b[1:] = i_d[0:-1] - C * delta_x / delta_t * (v_c[0:-1] - v_a[0:-1])
    v_a[1:] = v_c[0:-1] - L * delta_x / delta_t * (i_d[0:-1] - i_b[0:-1])

    # i_d = i_b - delta_x / (vp * z0 * delta_t) * (v_c - v_a)
    v_a = v_c.copy()
    i_b = i_d.copy()

    # v_c = stimulus(x / c + 500 * ps)
    update_plot(delta_x * np.array(range(0, num_points)), v_c)
    plt.pause(0.01)
    print(time)

