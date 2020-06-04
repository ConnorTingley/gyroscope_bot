import numpy as np
from numpy import random
import matplotlib.pyplot as plt

class disturbance:
    mus = [0,0,0] #mean for time between disturbances, duration of disturbance, and force
    jerk = [0, 0, 0] #left, middle right value for jerk
    sigmas = [0,0,0] #variances for above
    max_impulse = 0
    jerk = 0

    def __init__(self, averages = [1, 1, 0], spreads = [1, 1, 1], max = 1, jerk_mean = 1, jerk_spread = .5):
        self.mus = averages
        self.sigmas = spreads
        self.jerk = [jerk_mean - jerk_spread/2, jerk_mean, jerk_mean + jerk_spread/2]
        max_impulse = max

    def generate_events(self, start_time, sim_len):
        t = start_time
        count = 0
        impulse = np.inf
        delta_t = 0
        prev_max_time = 0
        events = []
        while t < sim_len:
            #draw new delays until we are satisfied
            while prev_max_time > delta_t:
                delta_t = random.normal(self.mus[0], self.sigmas[0])
            #draw new impulses until we are satisfied
            while impulse > self.max_impulse:
                fs = random.normal(self.mus[2], self.sigmas[2], 3)
                durs = random.normal(self.mus[1], self.sigmas[1], 3)
                durs = np.clip(durs, a_min = 0, a_max = None)
                js = random.triangular(self.jerk[0], self.jerk[1], self.jerk[2], 3)
                prev_max_time = np.max(durs + 2 * np.divide(fs, js))
                impulse = fs.dot(durs)
            t += delta_t
            events.append([t, fs, js, durs])
            impulse = np.inf
        for i in range(len(events)):
            print(i, events[i][0])
        return events

    def plot_events(self, events, dt):
        t = 0
        count = 0
        event_done = False
        t_arr = []
        f_arr = [[],[],[]]
        while count < len(events):
            new_t = events[count][0]
            no_f = np.arange(t, new_t, step = dt)
            t_arr.extend(no_f)
            f_append = [[],[],[]]
            for i in range(3):
                f_arr[i].extend(np.zeros(len(no_f)))
                f = 0
                df = dt * events[count][2][i]
                while (f + df) < events[count][1][i]:
                    f += df
                    f_append[i].append(f)
                f_append[i].extend(np.ones(int(events[count][3][i] / dt)) * events[count][1][i])
                while (f - df) > 0:
                    f -= df
                    f_append[i].append(f)
            max_added = np.max([len(f_append[0]), len(f_append[1]), len(f_append[2])])
            for i in range(3):
                f_append[i].extend(np.zeros(max_added - len(f_append[i])))
                f_arr[i].extend(f_append[i])
            t_arr.extend(np.arange(0, max_added) * dt + new_t)
            t = t_arr[len(t_arr) - 1]
            count += 1
        print(f_arr)
        plt.plot(t_arr, f_arr[0], label = "f_x")
        plt.plot(t_arr, f_arr[1], label = "f_y")
        plt.plot(t_arr, f_arr[2], label = "f_z")
        plt.xlabel("t")
        plt.legend()
        plt.savefig("test.png")