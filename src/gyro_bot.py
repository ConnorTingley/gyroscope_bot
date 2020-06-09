from simulate import state
from disturbances import disturbance
import matplotlib.pyplot as plt
import numpy as np

def main():
    disturbances = disturbance(averages = [2, 1, 0], spreads = [.25, .25, .125], max = 1, jerk_mean = 3, jerk_spread = 1)

    #events = disturbances.generate_events(1, 10)
    #disturbances.plot_events(events, .05)

    sim = state(mass = 10, length = 1, g=1, L2 = .25, events = [[1, np.array([0,0,0]), np.array([1,1,1]), np.array([1,1,1])]], max_flywheel_l= np.array([1.,1.,1.]), max_torque = np.array([1.,1.,1.]))
    num = 300
    angles = np.zeros((3, num))
    t = np.zeros(num)
    for i in range(num):
        new = sim.rough_step(.0003, np.array([0,0,0]), 300)
        t[i] = new[0]
        angles[:, i] = new[1]
    plt.scatter(t, angles[0,:], label = "theta")
    print(np.max(angles[1,:]))
    plt.scatter(t, angles[1,:], label = "phi")
    plt.xlabel("t")
    plt.ylabel("angle")
    plt.legend()
    plt.savefig('./test.png')
if __name__ == '__main__':
    main()