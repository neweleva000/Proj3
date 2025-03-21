from math import ceil
import numpy as np
import matplotlib.pyplot as plt

#environment constants
fs = 1E-15
c = 3E8

def stimulus(t, delta_t):
    tau = 30 * delta_t
    t0 = 3 * tau
    return np.exp(-((t - t0) / tau)**2)

#begin = True -> abc at begining of line
#begin = False -> abc at end of line
#Returns v
def absorbative_bc(begin, v_a):
    if begin:
        return v_a[1]
    else:
        return v_a[-2]

# Update function
def update_plot(new_x, new_y, ax):
    ax.cla() # Clear the current axes
    ax.plot(new_x, new_y) # Redraw with updated data
    plt.draw()

class Boundary_condition:
    #If the boundary is defined by a closed from soln.
    function_source = True
    function = None

    #If is absorbative_bc
    abc = False

    #If the boundary is another material
    tline_conn = False
    beg_tline = None
    end_tline = None
   
    def __init__(self, function_source, function, \
            abc, beg_tline=None, end_tline=None):
        self.function_source = function_source
        self.function = function
        self.beg_tline = beg_tline
        self.end_tline = end_tline
        self.abc = abc
    
    def gen_condition(self,time, dt, dx, begin, v_a):
        if self.function_source:
            return self.function(time, dt)
        if self.abc:
            return self.function(begin, v_a)
        else:
            pass
            #return tline_boundary(args[2],args[3],args[4])

    #TODO this is not completed
    #Returns [ [vbeg, vend], [ibeg, iend]]
    def tline_boundary(time, dx, dt):
        #copy update eq for a point
        i_beg_prev = beg_tline.i_b
        v_a = 1
        C = 1
        L = 1


class Transmission_line:
    #Interface between boundary conditions
    #Should be callable and return boundary values
    begin = None
    end = None

    #Tline parameters
    z0 = 0
    G = 0
    L = 0
    R = 0
    C = 0
    Er = 1
    vp = c/Er
    length = 0
    num_points = 0


    #Arrays representing values as a function of position
    #Current time step positional array
    v_a = None
    i_b = None
    #Next time step positional array
    v_c = None
    i_d = None

    #TODO account for non-lossless
    def __init__(self, z0, Er, begin, end, length, dx, lossless=True):
        self.z0 = z0
        self.Er = Er
        self.vp = c/Er
        self.L = z0 / self.vp
        self.C = 1/(z0 * self.vp)
        self.begin = begin
        self.end = end
        self.length = length
        self.num_points = ceil(length / dx)

        self.v_a = np.zeros(self.num_points)
        self.i_b = np.zeros(self.num_points)
        self.v_c = np.zeros(self.num_points)
        self.i_d = np.zeros(self.num_points)
    
    #TODO add non lossless update components
    def update_t_line(self, time, dx, dt):
        
        # Calc next time step
        self.i_d[0:-1] = self.i_b[0:-1] -\
                dt / (self.L * dx) *\
                (self.v_a[1:] - self.v_a[0:-1])

        self.v_c[0] = self.begin.gen_condition(time,dt,\
                dx, True, self.v_a)
        #self.v_c[0] += self.v_a[1]

        self.v_c[1:] = self.v_a[1:] - \
                dt / (self.C * dx) *\
                (self.i_d[1:] - self.i_d[0:-1])


        #TODO this isn't working for some reason
        self.v_c[-1] = self.i_b[-2] * self.z0
        #self.end.gen_condition(time,dt,\
        #dx, False, self.v_a)#stimulus(time, dt)

        #TODO this doesn't get used in current implementation don't know purpose
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
    delta_t = 5 * fs  
    delta_x = c * delta_t * 2
    num_points = 100
    length = delta_x * num_points
    beg = Boundary_condition(True, stimulus, False)
    end = Boundary_condition(False, absorbative_bc, \
            True)
    tline = Transmission_line(z0, Er, beg, end, length,\
            delta_x, is_lossles)

    #Simulation parameters
    num_cycles = 1000
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    x_dim = delta_x * np.array(range(0, num_points))
    line, = ax.plot(x_dim, tline.v_c)


    #Iterate and plot through time cycle
    for n in range(0, num_cycles):
        time = n * delta_t
        
        tline.update_t_line(time, delta_x, delta_t)

        update_plot(x_dim, tline.v_a, ax)
        plt.pause(0.01)
        print(time)
main()
