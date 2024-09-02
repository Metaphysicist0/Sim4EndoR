from SofaGW import SimController, example_vessel
from SofaGW.utils import SaveImage

import logging
import gym
import numpy as np
from gym.utils import seeding
from gym.spaces import Box
import math
from tqdm import tqdm
logger = logging.getLogger(__name__)

def get_dist(a, b):
    return np.sum((a-b)**2)

class CustomEnv(gym.Env):

    def __init__(self):
        self.state = 1
        self.aim = np.array([1,10,100])
        # 在这里改地图
        example_vessel = 'G:\\v24.06.00\\SofaGuidewireNav-main\\SofaGW\\vessel\\phantom.obj'
        self.vessel_filename = example_vessel
        self.sim = SimController(timeout=100, vessel_filename=example_vessel)
        self.action_space = Box(low=np.array([0.0, -4.0]), high=np.array([4.0, 4.0]))
        self.observation_space = Box(low=np.array([0.0, 0.0]), high=np.array([100.0, 100.0]))

        # Open the file in write mode, overwrite the existing file
        self.action_file = open('actions.txt', 'w')

    def get_obs(self):
        position = self.sim.get_GW_position().reshape(1,-1)
        velocity = self.sim.get_GW_velocity().reshape(1,-1)

        #print(position.shape, velocity.shape)
        #print(position.min(axis=0),position.max(axis=0), velocity.min(axis=0),velocity.max(axis=0))
        return np.concatenate((position, velocity), axis=1)

    def reset(self):
        self.sim.reset()
        self.step_count = 0
        obs = self.get_obs()
        #print(obs)
        self.last_position = obs[0, :3]

        return obs[0].reshape(-1,), {}

    def step(self, action):
        self.sim.action(translation=action[0], rotation=action[1])
        err = self.sim.step()

        # Save action to file
        self.action_file.write(f'{action[0]}, {action[1]}\n')

        reward = 0
        Done = err
        if not Done:
            state = self.get_obs()
            last_dist = get_dist(self.last_position, self.aim)
            cur_pos = state[0, :3]
            cur_dist = get_dist(cur_pos, self.aim)
            reward = (last_dist - cur_dist)/ 100.0
            if cur_dist <= 1:
                reward += 100
                Done = True

            self.last_position = cur_pos
            #print(self.last_position)
        else:
            state = np.array([0,0,0,0,0,0]).reshape(1,-1)
            reward = -100
        Truncate = False
        info = {}
        self.step_count += 1
        if self.step_count >= 100:
            Done = True
        print('r', reward)
        print('state', state[0].reshape(-1,))

        #print(state[-1].reshape(-1).shape)
        return state[0].reshape(-1,), reward, Done, Truncate, info

    def render(self):
        image = self.sim.GetImage()
        SaveImage(image=image, filename=f'image/image_{self.step_count}.jpg')

    def close(self):
        # Close the action file when the environment is closed
        self.action_file.close()


if __name__ == "__main__":
    env = CustomEnv()
    env.reset()
    for t in tqdm(range(400)):
        # 随机选择一个动作
        action = env.action_space.sample()
        # 执行动作 获取环境反馈
        observation, reward, done, _, info = env.step(action)

        # 如果玩死了就退出
        if done:
            break
        env.render()
    env.close()  # Ensure the file is closed properly