from SofaGW import SimController, example_vessel
from SofaGW.utils import SaveImage

import logging
import gym
import numpy as np
from gym.utils import seeding
from gym.spaces import Box
import math
logger = logging.getLogger(__name__)


class CustomEnv(gym.Env):

    def __init__(self):
        self.state = 0
        self.vessel_filename = example_vessel
        self.sim = SimController(timeout=1000000, vessel_filename=example_vessel)
        self.action_space = Box(low=np.array([0.0, 0.0]), high=np.array([100.0, 100.0]))
        self.observation_space = Box(low=np.array([0.0, 0.0]), high=np.array([100.0, 100.0]))

    def get_obs(self):
        position = self.sim.get_GW_position()
        velocity = self.sim.get_GW_velocity()
        print(position, velocity)
        return np.concatenate((position, velocity))

    def reset(self):
        self.sim.reset()
        return self.get_obs(), {}

    def step(self, action):
        self.sim.action(translation=action[0], rotation=action[1])
        err = self.sim.step()
        state = self.get_obs()
        reward = 0
        Done = err
        Truncate = False
        info = {}
        return state, reward, Done, Truncate, info

    def render(self):
        image = self.sim.GetImage()
        SaveImage(image=image, filename=f'image/image_{i}.jpg')


if __name__ == "__main__":
    env = CustomEnv()
    env.reset()()
    for t in range(100):
        # 随机选择一个动作
        action = env.action_space.sample()
        # 执行动作 获取环境反馈
        observation, reward, done, info = env.step(action)
        # 如果玩死了就退出
        if done:
            break
        env.render()
