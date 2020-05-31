import numpy as np
class state:
    mass = 0
    length = 0
    position = np.empty([3])
    omega = np.empty([3])
    wheel_ws = np.empty([3])
    I = 0
    def __init__(self, envelope, mass = 10, length = 1,):
        self.mass = mass
        self.length = length
        self.I = mass * length * length
        self.theta = np.empty([3])
        self.omega = np.empty([3])
        self.envelope = envelope
        self.H = np.empty([3])

    def simulate_step(self, dt):










