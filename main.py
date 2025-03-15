import numpy as np

import matplotlib.pyplot as plt

#environment constants
fs = 1E-15
c = 3E8
num_cycles = 1000
num_points = 100

def stimulus(t, delta_t):
    tau = 30 * delta_t
    t0 = 3 * tau
    return np.exp(-((t - t0) / tau)**2)

# Update function
def update_plot(new_x, new_y, ax):
    ax.cla() # Clear the current axes
    ax.plot(new_x, new_y) # Redraw with updated data
    plt.draw()

class transmission_line:
    z0 = 0
    G = 0
    L = 0
    R = 0
    C = 0
    Er = 1
    vp = c/Er
    dt = 0
    dx = 0

    #Arrays representing values as a function of position
    #Previous time step positional array
    v_a = np.zeros(num_points)
    i_b = np.zeros(num_points)
    #Next time step positional array
    v_c = np.zeros(num_points)
    i_d = np.zeros(num_points)

    #TODO account for non-lossless
    def __init__(self, z0, Er, dx, dt, lossless=True):
        self.z0 = z0
        self.Er = Er
        self.vp = c/Er
        self.L = z0 / self.vp
        self.C = 1/(z0 * self.vp)
        self.dx = dx
        self.dt = dt
    
    #TODO add non lossless update components
    def update_t_line(self, time):
        self.v_c[0] = stimulus(time, self.dt)
        
        # step 1, time step
        self.v_c[1:] = self.v_a[1:] - \
                self.dt / (self.C * self.dx) *\
                (self.i_b[1:] - self.i_b[0:-1])
        self.i_d[0:-1] = self.i_b[0:-1] -\
                self.dt / (self.L * self.dx) *\
                (self.v_c[1:] - self.v_c[0:-1])

        #TODO this doesn't get used in current implementation
        # step 2, space step
        #i_b[1:] = i_d[0:-1] - C * delta_x / delta_t *\
        #        (v_c[0:-1] - v_a[0:-1])
        #v_a[1:] = v_c[0:-1] - L * delta_x / delta_t *\
        #        (i_d[0:-1] - i_b[0:-1])

        #Store generated array
        self.v_a = self.v_c.copy()
        self.i_b = self.i_d.copy()


def main():
    #Tline characteristics 
    z0 = 50
    Er = 1
    is_lossles = True 
    delta_t = 10 * fs  
    delta_x = c * delta_t * 2

    tline = transmission_line(z0, Er, delta_x, delta_t,\
            is_lossles)

    #Simulation parameters
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    x_dim = delta_x * np.array(range(0, num_points))
    line, = ax.plot(x_dim, tline.v_c)


    #Iterate and plot through time cycle
    for n in range(0, num_cycles):
        time = n * delta_t
        
        tline.update_t_line(time)

        update_plot(x_dim, tline.v_c, ax)
        plt.pause(0.01)
        print(time)
main()
