import numpy as np
import matplotlib.pyplot as plt

import csv


def get_logged_data(dtr):
    with open('dispersion-{}.csv'.format(dtr), newline='') as csvfile:
        freq = []
        arg = []
        reader = csv.reader(csvfile, delimiter=',', dialect='excel')
        for row in reader:
            freq.append(float(row[0].strip('[').rstrip(']')))
            arg.append(float(row[1].strip('[').rstrip(']')))

    return np.array(freq), np.array(arg)


def main():
    for dtr in [5, 10, 20, 50]:
        freq, arg = get_logged_data(dtr)
        print(freq)
        # return
        plt.plot(freq, arg, label='$\\Delta t = {:.02} ps$'.format(dtr/100))
    plt.legend()
    plt.xlim([0, 50])
    plt.ylim([-0.1, 0.1])
    plt.ylabel('Phase error (rad)')
    plt.xlabel('Frequency (GHz)')
    plt.title('Dispersion at different timestep sizes for $\\tau = 30 \\Delta t$')
    plt.show()


main()