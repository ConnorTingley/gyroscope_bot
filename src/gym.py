import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from simulate import state


class PendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.dt = .0003
        self.step_count = 300
        self.max_angular_vel = 1
        self.max_torque = 1.
        self.max_flywheel = 1.
        self.g = 9.8
        self.mass = 10
        self.length = 1.
        self.L2 = 0.25
        self.death_pos = np.pi/2
        self.viewer = None

        self.sim = state(mass=self.mass, length=self.length, g=self.g, L2=self.L2,
                    events=[[1, np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([1, 1, 1])]],
                    max_flywheel_l=np.array([self.max_flywheel, self.max_flywheel, self.max_flywheel]), max_torque=np.array([self.max_torque, self.max_torque, self.max_torque]))
        self.state = self.sim.rough_step(self.dt, [0,0,0], 0)

        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1]),
            high=np.array([1, 1, 1]),
            shape=(3,),
            dtype=np.float32
        )
        #Observation Space formatting: [phi, sin(alpha), cos(alpha), facing, d_phi, d_alpha, d_facing, wheel_l1, wheel_l2, wheel_l3]
        self.observation_space = spaces.Box(
            low=np.array([0, -1., -1., , -self.max_angular_vel, -self.max_angular_vel, -self.max_angular_vel, -1, -1, -1], dtype=np.float32),
            high=np.array([self.death_pos, 1., 1., , self.max_angular_vel, self.max_angular_vel, self.max_angular_vel, 1, 1, 1], dtype=np.float32),
            shape=(9,),
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        self.state = self.sim.rough_step(self.dt, u, self.step_count), reward


        return self._get_obs(), reward, False, {}

    def reset(self):
        self.state = self.sim.reset()
        return self._get_obs()

    def _get_obs(self):
        t, angle, angle_vel, wheel_l = self.state
        theta, phi, alpha, facing = angle
        d_theta, d_alpha, d_phi, d_facing = angle_vel

        return np.array([phi, np.sin(alpha), np.cos(alpha), facing, d_phi, d_alpha, d_facing, wheel_l[0], wheel_l[1], wheel_l[2]])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

            rod = rendering.Line((0,0), (0,1))
            rod.set_color(0., 0., 0.)

            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)

            tip = rendering.make_circle(.1)
            tip.set_color(0.2, 0.2, 0.9)
            self.tip_transform = rendering.Transform()
            rod.add_attr(self.tip_transform)
            self.viewer.add_geom(tip)


        self.viewer.add_onetime(self.img)
        angle = self.state[0]
        radius = np.sin(angle[1])
        self.pole_transform.set_rotation(angle[0])
        self.pole_transform.set_scale(1,radius)

        self.tip_transform.set_translation(radius * np.cos(angle),radius * np.sin(angle))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)