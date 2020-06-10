from simulate import state
from disturbances import disturbance
import matplotlib.pyplot as plt
import numpy as np

def main():
    disturbances = disturbance(averages = [2, 1, 0], spreads = [.25, .25, .125], max = 1, jerk_mean = 3, jerk_spread = 1)

    #events = disturbances.generate_events(1, 10)
    #disturbances.plot_events(events, .05)

    sim = state(mass = 10, length = 1, g=0, L2 = .25, events = [[1, np.array([0,0,0]), np.array([1,1,1]), np.array([1,1,1])]], max_flywheel_l= np.array([1.,1.,1.]), max_torque = np.array([1.,1.,1.]))
    num = 300
    angles = np.zeros((3, num))
    angle_vels = np.zeros((3, num))
    t = np.zeros(num)
    for i in range(num):
        new = sim.rough_step(.03, np.array([0,0,0]), 20)
        t[i] = new[0]
        angles[:, i] = new[1]
        angle_vels[:, i] = new[2]
    plt.scatter(t, angle_vels[0,:], label = "theta_dot")
    plt.scatter(t, angle_vels[1, :], label="phi_dot")
    print(np.max(angles[1,:]))
    plt.scatter(t, angles[2,:], label = "alpha")
    plt.scatter(t, angles[1,:], label = "phi")
    plt.xlabel("t")
    plt.ylabel("angle")
    plt.legend()
    plt.savefig('./test.png')
if __name__ == '__main__':
    main()