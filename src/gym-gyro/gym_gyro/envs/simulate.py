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
    angle = np.zeros(4) #represented as theta, phi, alpha, facing (note: theta doesn't actually matter--symmetry yo)
    angle_vel = np.zeros(4) #theta_dot, phi_dot, alpha_dot, facing
    wheel_l = np.zeros(3) #momentum in flywheels

    #for backprop
    reward_consts = [10., 4., 0.]


    events = []
    def __init__(self, mass = 10, length = 1, g=10, L2 = .25,
                 events = [], max_flywheel_l= np.array([1,1,1]), max_torque = np.array([1,1,1])):
        #physical parameters of system
        self.I = np.array([mass * length * length, mass * length * length, mass * L2 * L2])
        self.ag = g * mass / self.I[0]
        self.satr_l = max_flywheel_l
        self.max_torque = max_torque

        #initial position / velocity
        #self.angle = np.array([0., .2, 0.])
        self.angle = np.array([0.,np.random.triangular(.1, .2, .3),0., 0.])
        self.angle_vel = np.array([.1,0.,0., 0.])
        self.wheel_l = np.array([0.,0.,0.])
        self.t = 0
        self.event_num = 0
        self.cur_event = events[0]
        self.event_end = self.cur_event[0] + np.max(self.cur_event[3])

        #events which will occur
        self.events = events



    def rough_step(self, dt, T_action, count):
        for i in range(count):
            # convert momentum space command to wheel command
            wheel_action = np.zeros(3)
            transform = np.array([[np.cos(self.angle[2]), -np.sin(self.angle[2]), 0],
                         [np.sin(self.angle[2]), np.cos(self.angle[2]), 0],
                         [0,0,1]])
            wheel_action = transform.dot(T_action) * np.array([np.sqrt(2), np.sqrt(2), 1]) * self.max_torque

            #check bounding on momentum
            upper_torque_lim = self.max_torque * (np.ones(3) - self.wheel_l / self.satr_l)
            lower_torque_lim = -self.max_torque * (np.ones(3) + self.wheel_l / self.satr_l)
            wheel_action_clipped = np.clip(wheel_action, lower_torque_lim, upper_torque_lim)
            self.wheel_l += wheel_action_clipped * dt

            #compute angular accelerations from wheels
            applied_T = np.transpose(transform).dot(wheel_action_clipped)
            a = applied_T / self.I
            a[1] += self.ag * np.sin(self.angle[1])

            #determine accelerations from events
            T_inc = (self.t - self.cur_event[0]) * self.cur_event[2]
            T_steady = self.cur_event[1]
            T_dec = (self.cur_event[0] + self.cur_event[3] - self.t) * self.cur_event[2]
            a += [max(0, min(*l)) for l in zip(T_inc, T_steady, T_dec)] / self.I

            #check if we have concluded this event
            if self.t > self.event_end:
                self.event_num += 1
                if self.event_num < len(self.events):
                    self.cur_event = self.events[self.event_num]
                    self.event_end = self.cur_event[0] + max(self.cur_event[3])

            #velocity updates
            #runge kutta changes
            new_theta_dot = self.angle_vel[0] * (1 - 2 * self.angle_vel[1] * dt / np.tan(self.angle[1])) \
                                + a[0] * dt
            new_angle = self.angle + self.angle_vel * dt
            #self.angle_vel[0] = self.angle_vel[0] * (1 + 2 * self.angle_vel[1] * dt / np.tan(self.angle[1])) \
            #                    + a[0] * dt
            #self.angle_vel[1] +=  a[1] * dt \
            #                    + np.sin(2 * self.angle[1]) * (self.angle_vel[0]) ** 2 * dt / 2

            self.angle_vel[1] +=  a[1] * dt \
                                + np.sin(new_angle[1] + self.angle[1]) * ((new_theta_dot + self.angle_vel[0]) / 2) ** 2 * dt / 2
            self.angle_vel[0] = new_theta_dot
            self.angle_vel[2] += a[2] * dt
            self.angle_vel = np.clip(-1, self.angle_vel, 1)

            #position updates (Precise step: 1st (2nd?) order runge kutta)
            self.angle += self.angle_vel * dt
            self.angle[2] = 0
            #if we have crossed over vertical, need to flip some coordinates & velocities
            if self.angle[1] < 0:
                self.angle[0] = -self.angle[0]
                self.angle[1] = -self.angle[1]
                self.angle_vel[1] = -self.angle_vel[1]
                self.angle[2] = -self.angle[2]

            if self.angle[2] > np.pi:
                self.angle[2] -= 2 * np.pi
            if self.angle[2] < -np.pi:
                self.angle[2] += 2 * np.pi

            self.t += dt
        fractional_L = self.wheel_l / self.satr_l
        phi_dot_opt = - self.angle[1] / np.sqrt(self.ag)
        reward = - self.reward_consts[0] * np.cos(self.angle[1]) - \
                 self.reward_consts[1] * fractional_L[1] ** 2 - self.reward_consts[2] * (self.angle_vel[1] - phi_dot_opt)**2
        return [self.t, self.angle, self.angle_vel, self.wheel_l/ self.satr_l], reward

    def reset(self):
        self.angle = np.array([0.,np.random.triangular(.3, .2, .1),0.,0.])
        self.angle_vel = np.array([.0,0.,0.,0.])
        self.wheel_l = np.array([0.,0.,0.])
        self.t = 0

        self.event_num = 0
        self.cur_event = self.events[0]
        self.event_end = self.cur_event[0] + np.max(self.cur_event[3])

        return [self.t, self.angle, self.angle_vel, self.wheel_l / self.satr_l]