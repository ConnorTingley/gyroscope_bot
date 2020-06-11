from gym.envs.registration import register

register(
    id='gyro-v0',
    entry_point='gym_gyro.envs:GyroEnv',
)