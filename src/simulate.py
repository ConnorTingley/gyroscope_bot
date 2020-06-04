from __future__ import division
import numpy as np
import math

class state:
    # physical parameters of system
    I = np.zeros(3)
    ag = 0
    max_torque = np.zeros(3)
    satr_l = np.zeros(3)

    #state variables
    t = 0
    event_num = 0
    event_end = 0
    cur_event = []
    angle = np.zeros(3) #represented as theta, phi, alpha (note: theta doesn't actually matter--symmetry yo)
    angle_vel = np.zeros(3) #theta_dot, phi_dot, alpha_dot
    wheel_l = np.zeros(3) #momentum in flywheels

    #for backprop
    t_overshoot = []


    events = []
    def __init__(self, envelope, mass = 10, length = 1, g=10, L2 = .25, events = [], max_flywheel_l= [], max_torque = []):
        #physical parameters of system
        self.I = [mass * length * length, mass * length * length, mass * L2 * L2]
        self.ag = g * mass / self.I
        self.satr_l = max_flywheel_l
        self.max_torque = max_torque

        #initial position / velocity
        self.angle = [0,0,0]
        self.angle_vel = [0,0,0]
        self.wheel_l = [0,0,0]
        self.t = 0
        event_num = 0
        cur_event = events[0]
        event_end = cur_event[0] + max(cur_event[3])

        #events which will occur
        self.events = events



    def rough_step(self, dt, T_action, count):
        torque_overshoot = np.zeros(3)
        for i in range(count):
            # convert momentum space command to wheel command
            wheel_action = np.zeros(3)
            transform = [[np.cos(self.angle[2]), -np.sin(self.angle[2]), 0],
                         [np.sin(self.angle[2]), np.cos(self.angle[2]), 0],
                         [0,0,1]]
            wheel_action = transform.dot(T_action)

            #check bounding on momentum
            upper_torque_lim = self.max_torque * (np.ones(3) - self.wheel_l / self.satr_l)
            lower_torque_lim = -self.max_torque * (np.ones(3) + self.wheel_l / self.satr_l)
            wheel_action_clipped = np.clip(wheel_action, lower_torque_lim, upper_torque_lim)
            torque_overshoot += wheel_action - wheel_action_clipped
            self.wheel_l += wheel_action_clipped * dt

            #compute angular accelerations from wheels
            applied_T = np.transpose(transform).dot(wheel_action_clipped)
            a = applied_T / self.I
            a[1] += self.ag * np.sin(self.angle[1])

            #determine accelerations from events
            T_inc = (self.t - self.cur_event[0]) * self.self.cur_event[2]
            T_steady = self.cur_event[1]
            T_dec = (self.cur_event[0] + self.events[self.event_num][3] - self.t) * self.events[self.event_num][2]
            a += [max(0, min(*l)) for l in zip(T_inc, T_steady, T_dec)] / self.I

            #check if we have concluded this event
            if self.t > self.event_end:
                self.event_num += 1
                self.cur_event = self.events[self.event_num]
                self.event_end = self.cur_event[0] + max(self.cur_event[3])

            #velocity updates
            self.angle_vel[0] = (1 + 2 * self.angle_vel[1] * dt / np.tan(self.angle[1])) * self.angle_vel[0] \
                                + a[0] * dt
            self.angle_vel[1] = self.angle_vel[1] + a[1] * dt \
                                + np.sin(2 * self.angle[1]) * self.angle_vel[0] ** 2 * dt / 2
            self.angle_vel[2] += a[2] * dt

            #position updates (Precise step: 1st (2nd?) order runge kutta)
            self.angle = self.angle_vel * dt

            #if we have crossed over vertical, need to flip some coordinates & velocities
            if self.angle[1] < 0:
                self.angle[1] = -self.angle[1]
                self.angle_vel[1] = -self.angle_vel[1]
                self.angle[2] = -self.angle[2]

            self.t += dt
        torque_overshoot = torque_overshoot/count
        self.t_overshoot.append(torque_overshoot)
        return [self.angle[1:2], self.angle_vel, self.wheel_l/ self.satr_l]

    def precise_step(self, dt, action):
        return 3










