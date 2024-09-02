import argparse
import os
import sys
import random
import numpy as np
import gym
from env import CustomEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter
import logging

logging.basicConfig(level=logging.INFO,
                    filename='SAC.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)  # 'train' or 'test'
parser.add_argument("--env_name", default="Pendulum-v1", type=str)
parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)
parser.add_argument('--learning_rate', default=3e-4, type=float)  # SAC often uses a higher learning rate
parser.add_argument('--gamma', default=0.99, type=float)  # discounted factor
parser.add_argument('--capacity', default=1000000, type=int)  # replay buffer size, typically larger for SAC
parser.add_argument('--batch_size', default=256, type=int)  # SAC often uses larger batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
parser.add_argument('--alpha', default=0.2, type=float)  # Entropy regularization coefficient
parser.add_argument('--automatic_entropy_tuning', default=True, type=bool)  # Whether to automatically tune alpha
parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--render', default=False, type=bool)
parser.add_argument('--log_interval', default=50, type=int)
parser.add_argument('--load', default=False, type=bool)
parser.add_argument('--render_interval', default=100, type=int)
parser.add_argument('--max_episode', default=100000, type=int)
parser.add_argument('--update_interval', default=1, type=int)  # Update per step for SAC
parser.add_argument('--max_timesteps', default=1000, type=int)  # Max timesteps per episode
parser.add_argument('--save_model', default=True, type=bool)  # Whether to save the model
parser.add_argument('--load_model', default="", type=str)  # Model load filename

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
script_name = os.path.basename(__file__)
env = CustomEnv()

if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

directory = './exp' + script_name + args.env_name +'./'

class ReplayBuffer(object):
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SoftQNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Limit the range of log_std for stability
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()  # Reparameterization trick
        action = torch.tanh(z) * self.max_action
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class SAC(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)

        self.critic_1 = SoftQNetwork(state_dim, action_dim).to(device)
        self.critic_2 = SoftQNetwork(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
                                               lr=args.learning_rate)

        self.target_critic_1 = SoftQNetwork(state_dim, action_dim).to(device)
        self.target_critic_2 = SoftQNetwork(state_dim, action_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.alpha = args.alpha
        if args.automatic_entropy_tuning:
            self.target_entropy = -np.prod(env.action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.learning_rate)

        self.replay_buffer = ReplayBuffer()
        self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, _ = self.actor.sample(state)
        action = action.cpu().data.numpy().flatten()
        if evaluate:
            return action
        else:
            return (action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])).clip(
                env.action_space.low, env.action_space.high)

    def update(self):
        for _ in range(args.update_interval):
            state, next_state, action, reward, done = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(state).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).to(device)
            done = torch.FloatTensor(done).to(device)

            with torch.no_grad():
                next_action, next_log_prob = self.actor.sample(next_state)
                target_q1 = self.target_critic_1(next_state, next_action)
                target_q2 = self.target_critic_2(next_state, next_action)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
                target_q = reward + ((1 - done) * args.gamma * target_q).detach()

            current_q1 = self.critic_1(state, action)
            current_q2 = self.critic_2(state, action)# 计算Critic的损失
        critic_loss_1 = F.mse_loss(current_q1, target_q)
        critic_loss_2 = F.mse_loss(current_q2, target_q)
        critic_loss = critic_loss_1 + critic_loss_2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.writer.add_scalar('Loss/critic_loss', critic_loss.item(), global_step=self.num_critic_update_iteration)

        # 计算Actor的损失
        new_action, log_prob = self.actor.sample(state)
        q1_new_policy = self.critic_1(state, new_action)
        q2_new_policy = self.critic_2(state, new_action)
        q_new_policy = torch.min(q1_new_policy, q2_new_policy)

        actor_loss = (self.alpha * log_prob - q_new_policy).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.writer.add_scalar('Loss/actor_loss', actor_loss.item(), global_step=self.num_actor_update_iteration)

        # 自动调整熵系数
        if args.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()
            self.writer.add_scalar('Loss/alpha_loss', alpha_loss.item(), global_step=self.num_training)
            self.writer.add_scalar('Alpha', self.alpha, global_step=self.num_training)

        # 更新目标网络
        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        self.num_critic_update_iteration += 1
        self.num_actor_update_iteration += 1
        self.num_training += 1

def save(self):
    torch.save(self.actor.state_dict(), directory + 'actor.pth')
    torch.save(self.critic_1.state_dict(), directory + 'critic_1.pth')
    torch.save(self.critic_2.state_dict(), directory + 'critic_2.pth')
    torch.save(self.target_critic_1.state_dict(), directory + 'target_critic_1.pth')
    torch.save(self.target_critic_2.state_dict(), directory + 'target_critic_2.pth')
    if args.automatic_entropy_tuning:
        torch.save(self.log_alpha, directory + 'log_alpha.pth')

def load(self):
    self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
    self.critic_1.load_state_dict(torch.load(directory + 'critic_1.pth'))
    self.critic_2.load_state_dict(torch.load(directory + 'critic_2.pth'))
    self.target_critic_1.load_state_dict(torch.load(directory + 'target_critic_1.pth'))
    self.target_critic_2.load_state_dict(torch.load(directory + 'target_critic_2.pth'))
    if args.automatic_entropy_tuning:
        self.log_alpha = torch.load(directory + 'log_alpha.pth')
    print("====================================")
    print("model has been loaded...")
    print("====================================")

def main():
    agent = SAC(state_dim, action_dim, max_action)
    if args.load_model:
        agent.load()

    total_step = 0
    for i in range(args.max_episode):
        state = env.reset()
        if isinstance(state, tuple):  # 如果 env.reset() 返回的是一个元组
            state = state[0]  # 获取第一个元素作为状态
        print(f"Initial state shape: {state.shape}")

        episode_reward = 0
        for t in range(args.max_timesteps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            if isinstance(next_state, tuple):  # 如果 env.step() 返回的是一个元组
                next_state = next_state[0]  # 获取第一个元素作为状态
            print(f"Next state shape: {next_state.shape}")

            agent.replay_buffer.push((state, next_state, action, reward, done))

            state = next_state
            episode_reward += reward

            if total_step % args.update_interval == 0:
                agent.update()

            if done:
                break

            total_step += 1

        print("Episode: {}, Total Reward: {:.2f}".format(i, episode_reward))

        if i % args.log_interval == 0 and args.save_model:
            agent.save()

        if args.render and i % args.render_interval == 0:
            env.render()

if __name__ == '__main__':
    main()