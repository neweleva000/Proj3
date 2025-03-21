import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt
from math import ceil
from math import sqrt
from numpy import log
import csv

ps = 1E-12
c = 3E8

delta_t = 0.5 * ps  # time step
delta_x = c * delta_t * 2   #cm

# these need to be global in order to extract dispersion
tau = 30 * delta_t  # 30
t0 = 3 * tau


def stimulus(t):
    return np.exp(-((t - t0) / tau) ** 2)


#stimulus_index = 20  # round(num_points / 2)


def calc_i_next(i_prev, v_now, L):
    return i_prev - (delta_t / (L * delta_x)) * (v_now[1:] - v_now[0:-1])


def calc_v_next(v_prev, i_now, v_next_left, v_next_right, C):
    result = v_prev[1:-1] - (delta_t / (C * delta_x)) * (i_now[1:] - i_now[0:-1])
    return np.concatenate((np.array([v_next_left]),
                           result,
                           np.array([v_next_right])))


class TransmissionLine:
    def __init__(self, z0, Er, length_cm):
        self.z0 = z0   # characteristic impedance
        self.vp = c / np.sqrt(Er)  # propagation velocity
        self.L = self.z0 / self.vp  # pul inductance (H/m)
        self.C = 1 / (self.z0 * self.vp)  # cap pul  (F/m)
        # number of spatial steps for this line 
        self.num_dx = ceil(length_cm/delta_x / 100)

        # arrays for FDTD calculation
        self.v_p = np.zeros(self.num_dx)
        self.i_p = np.zeros(self.num_dx - 1)
        self.v_a = np.zeros(self.num_dx)
        self.i_b = np.zeros(self.num_dx - 1)
        self.v_c = np.zeros(self.num_dx)
        self.i_d = np.zeros(self.num_dx - 1)

    def transmitted_voltage_right(self):
        return 1 / 2 * (self.v_p[-2] + (self.i_d[-1] + self.i_b[-1]) / 2 * self.z0)

    def transmitted_voltage_left(self):
        return 1 / 2 * (self.v_p[1] - (self.i_d[0] + self.i_b[0]) / 2 * self.z0)


fast_forward_n = 25  # only plots every n iterations to speed up calculation


class Setup:
    def __init__(self):
        self.tline_chain = []
        self.sftf_source = None

    def config_sftf_measurement(self, stim_length, probe_length, stim_function, num_cycles):
        stim_index = ceil(stim_length / delta_x / 100)
        probe_index = ceil(probe_length / delta_x / 100)
        self.sftf_source = (stim_index, probe_index, stim_function, num_cycles)

    def add_tline_to_chain(self, tline):
        self.tline_chain.append(tline)

    def calc_freq_domain(self,refl_array, plt, stim_array,\
            probe_array1, probe_array2, i_probe,\
            probe_offset, begin_impedance, end_impedance):

        in_fft = np.fft.fft(stim_array)
        probe_fft1 = np.fft.fft(probe_array1)
        probe_fft2 = np.fft.fft(probe_array2)
        refl_fft = np.fft.fft(refl_array)
        i_probe_fft = np.fft.fft(i_probe)

        #indexing
        w_v = np.fft.fftfreq(len(probe_fft1), delta_t) / 1E9
        index_0 = np.argmin(np.abs(w_v))
        index_5 = np.argmin(np.abs(w_v - 5))

        plt.figure(3)
        plt.plot(w_v[index_0:index_5],np.abs(in_fft)[index_0:index_5], label='Stimulus')
        plt.plot(w_v[index_0:index_5],np.abs(probe_fft1)[index_0:index_5],label='Transmitted')
        plt.plot(w_v[index_0:index_5],np.abs(refl_fft)[index_0:index_5],label='Reflected')
        plt.title('Frequency Domain Voltage Signal')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Voltage')
        plt.legend()

        plt.figure(4)
        #Calc S11 0 to 5G
        input_voltage = in_fft
        refl_voltage = refl_fft
        s11 = np.abs(refl_voltage / input_voltage)
        p11 = np.abs(s11) **2
        plt.plot(w_v[index_0:index_5], p11[index_0:index_5])
        plt.title("Reflected Power")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Percent Power")

        
        #Reflected power
        input_power = sum(abs(in_fft) **2)
        refl_power = sum(abs(refl_fft) **2)
        R_p = (refl_power / input_power)
        print("Reflected power: " + str(R_p))

        plt.figure(5)
        #Calc S21 0 to 5G
        s21 = np.abs((probe_fft1/sqrt(end_impedance) ) / (input_voltage/sqrt(begin_impedance)))
        p21 = np.abs(s21) **2
        plt.plot(w_v[index_0:index_5], p21[index_0:index_5])
        plt.title("Transmit Power")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Percent Power")

        plt.figure(6)
        plt.plot(w_v[index_0:index_5], p21[index_0:index_5] + p11[index_0:index_5])

        #Transmitted power
        output_power = sum(abs(probe_fft1) **2)
        T_p = (output_power / end_impedance) / (input_power/begin_impedance)
        print("Transmitted power: " + str(T_p))

        print("Total power: " + str(T_p + R_p))

        #Calc Gamma     -- Adjust reflection to have variable location
        output_voltage2_mag = sum(abs(probe_fft2))

        plt.figure(7)
        gamma = (1/(probe_offset * delta_x) )* (log(probe_fft1/probe_fft2))
        freq = w_v
        normalized_beta = np.imag(gamma[index_0:index_5]) / (2 * np.pi * freq[index_0:index_5])
        plt.plot(w_v[index_0:index_5], np.real(gamma[index_0:index_5]),label="Alpha")
        # plt.plot(w_v[index_0:index_5], normalized_beta, label="Beta/(2 pi f)")
        plt.plot(w_v[index_0:index_5], np.imag(gamma[index_0:index_5]),label="Beta")
        plt.title("Propagation Constant")
        plt.xlabel("Frequency (Hz)")
        plt.legend()

        # grid dispersion
        plt.figure(8)
        freq_Hz = np.flip(np.fft.fftshift(w_v * 1E9))
        freq_GHz = freq_Hz / 1e9
        in_shifted = np.flip(np.fft.fftshift(in_fft))
        out_shifted = np.flip(np.fft.fftshift(probe_fft1))
        use_indices = np.argwhere((abs(in_shifted) > 0.01) * (freq_Hz > 0))
        unwrap_stim_distance = t0 * c
        unwrap_probe_distance = unwrap_stim_distance + 0.8 * 7.5 / 100
        unwrapped_stim = in_shifted * np.exp(1j * 2 * np.pi * freq_Hz * unwrap_stim_distance / c)
        arg_stim = np.unwrap(np.angle(unwrapped_stim[use_indices]))
        unwrapped_response = out_shifted * np.exp(1j * 2 * np.pi * freq_Hz * unwrap_probe_distance / c)
        arg_probe = np.unwrap(np.angle(unwrapped_response[use_indices]))
        # plt.plot(freq_GHz[use_indices], np.real(unwrapped_stim[use_indices]), 'o', label='Real(in)')
        # plt.plot(freq_GHz[use_indices], np.imag(unwrapped_stim[use_indices]), 'o', label='Imag(in)')
        filename_marker = int(100 * delta_t / ps)

        with open('dispersion-{}.csv'.format(filename_marker), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for i in range(0, len(arg_probe)):
                writer.writerow([freq_GHz[use_indices][i], arg_probe[i]])


        plt.plot(freq_GHz[use_indices], arg_stim, label='arg[in]')
        plt.plot(freq_GHz[use_indices], arg_probe, label='arg[out]')
        plt.ylabel('Phase deviation (radians)')
        plt.xlabel('Frequency (GHz)')
        plt.title('Grid dispersion for single 7.5cm line, $\\tau = {} \\Delta t$, $\\Delta t={:.02}$ ps'.format(
            int(tau/delta_t), delta_t/ps))
        # plt.plot(freq_GHz[use_indices], np.real(unwrapped_response[use_indices]), 'o', label='Imag(out)')
        # plt.plot(freq_GHz[use_indices], np.real(unwrapped_response[use_indices]), 'o', label='Real(out)')

        plt.legend()

        #Calc Characteristic Impedance --V / I_avg
        output_voltage1_mag = sum(abs(probe_fft1))
        z0 = output_voltage1_mag / sum(abs(i_probe_fft)) 
        print("Characteristic Impedance: " + str(z0))


    def run_sim(self):
        # extract SFTF/probe configuration from tuple
        probe_offset = 20 #TODO this was randomly chosen
        num_cycles = self.sftf_source[3]
        stim_function = self.sftf_source[2]
        probe_index = self.sftf_source[1]
        stim_index = self.sftf_source[0]
        total_points = sum(line.num_dx for line in self.tline_chain)
        stim_array = []
        probe_array1 = []
        probe_array2 = []
        i_probe = []
        refl_array = []


        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        x = np.array(range(0, total_points)) * delta_x
        line = ax.plot(x, np.zeros(total_points), label='Voltage')
        line2 = ax.plot(x, np.zeros(total_points),label='Current')
        ax.set_xlabel("Position (cm)")
        ax.set_title('Cascaded Transmission Line')
        ax.legend()

        # Update function
        def update_plot(new_x, new_y, new_x2, new_y2):
            ax.cla()  # Clear the current axes
            ax.plot(new_x * 100, new_y, label='Voltage')
            ax.plot(new_x2 * 100, new_y2,label='Current')  # Redraw with updated data
            ax.set_xlabel("Position (cm)")
            ax.set_ylabel("Voltage(V)/Current(A)")
            ax.set_title('Cascaded Transmission Line')
            ax.legend()
            
            # plt.ylim([-0.5, 1.1])
            plt.draw()

        for n in range(0, num_cycles):
            time = n * delta_t

            num_tlines = len(self.tline_chain)
            for idx, line in enumerate(self.tline_chain):

                feed_line = self.tline_chain[idx-1] if idx > 0 else None
                fed_line = self.tline_chain[idx+1] if idx < num_tlines - 1 else None

                # update voltage using central differencing
                if feed_line is None:
                    left_voltage = line.v_p[1]
                else:
                    left_gamma = (line.z0 - feed_line.z0) / (line.z0 + feed_line.z0)
                    from_line_wave = line.transmitted_voltage_left()
                    from_feed_wave = feed_line.transmitted_voltage_right()
                    left_voltage = ((1 + left_gamma) * from_feed_wave + (1 - left_gamma) * from_line_wave)

                if fed_line is None:
                    right_voltage = line.v_p[-2]
                else:
                    from_line_wave = line.transmitted_voltage_right()
                    from_fed_wave = fed_line.transmitted_voltage_left()
                    right_gamma = (fed_line.z0 - line.z0) / (fed_line.z0 + line.z0)
                    right_voltage = ((1 + right_gamma) * from_line_wave + (1 - right_gamma) * from_fed_wave)

                line.v_c = calc_v_next(line.v_a, line.i_b, left_voltage, right_voltage, line.C)

                if idx == 0:
                    # add SF/TF stimulus source
                    line.v_c[stim_index + 1] += stim_function(time) / 2
                    line.i_b[stim_index] += stim_function(time - delta_t / 2) / line.z0 / 2

                # update i from central differencing
                line.i_d = calc_i_next(line.i_b, line.v_c, line.L)

                # store last 3 voltages and currents for central differencing and absorbing BCs
                line.v_p = line.v_a.copy()
                line.v_a = line.v_c.copy()
                line.i_p = line.i_b.copy()
                line.i_b = line.i_d.copy()

            combined_voltage = np.concatenate(
                tuple(line.v_c for line in self.tline_chain)
            )
            combined_current_norm = np.concatenate(
                tuple(line.i_d * line.z0 for line in self.tline_chain)
            )
            combined_current = np.concatenate(
                tuple(line.i_d  for line in self.tline_chain)
            )
            combined_current_prev = np.concatenate(
                tuple(line.i_p  for line in self.tline_chain)
            )
            current_x = np.concatenate(
                tuple((np.arange(0, line.num_dx-1) + line.num_dx * idx) * delta_x + delta_x/2 for idx, line in enumerate(self.tline_chain))
            )
            stim_array.append(stim_function(time))
            refl_array.append(combined_voltage[stim_index - 2])
            probe_array1.append(combined_voltage[probe_index])
            probe_array2.append(combined_voltage[probe_index + probe_offset])

            #Current averaging probe
            i_avg = (combined_current[probe_index + 1] +\
                    combined_current_prev[probe_index+1] +\
                    combined_current_prev[probe_index] + \
                    combined_current[probe_index]) * 0.25
            i_probe.append(i_avg)

            if n % fast_forward_n == 0:
                update_plot(delta_x * np.array(range(0, total_points)), combined_voltage, current_x, combined_current_norm)
                plt.pause(0.01)
                print(time)

        plt.figure(2)
        time_plot = np.arange(0, num_cycles) * delta_t / ps
        plt.plot(time_plot, stim_array, time_plot, probe_array1, time_plot, refl_array)
        plt.legend(['Stimulus', 'Transmitted', 'Reflected'])
        plt.xlabel('Time (ps)')
        plt.ylabel('Voltage (V)')
        plt.title('Time-domain response')
 
        self.calc_freq_domain(refl_array, plt,stim_array,\
                probe_array1, probe_array2, i_probe,\
                probe_offset, self.tline_chain[0].z0, \
                self.tline_chain[-1].z0)
        plt.show(block=True)


def main():
    prop_length_sim = 1.5   # meters, distance a pulse will travel in the simulation time
    num_cycles = int(prop_length_sim / (delta_t * c))
    
    z0_1 = 50
    z0_2 = 100
    z0_3 = 50
    length1_cm = 7.5
    length2_cm = 7.5 
    length3_cm = 7.5

    tline1 = TransmissionLine(z0_1, 1, length1_cm)
    tline2 = TransmissionLine(z0_2, 1, length2_cm)
    tline3 = TransmissionLine(z0_3, 1, length3_cm)

    setup = Setup()

    stimulus_start_cm = 0.1 * length1_cm 
    probe_pos_cm = length1_cm + length2_cm + 0.9 * length3_cm
    #probe_pos_cm = length1_cm + 0.5 * length2_cm 
    # probe_pos_cm = 0.9 * length1_cm
    setup.add_tline_to_chain(tline1)
    setup.add_tline_to_chain(tline2)
    setup.add_tline_to_chain(tline3)
    setup.config_sftf_measurement(stimulus_start_cm, probe_pos_cm, stimulus, num_cycles)
    setup.run_sim()

main()

