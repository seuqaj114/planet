from gym.envs.registration import register

register(
    id='PendulumPixels-v0',
    entry_point='planet.scripts.envs.pendulum:PendulumPixelsEnv',
    max_episode_steps=50,
    reward_threshold=39)