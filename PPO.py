import argparse
import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="Pendulum-v1", type=str)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument("--clip_eps", default=0.2, type=float)  # Clipping parameter for PPO
parser.add_argument("--update_timestep", default=4000, type=int)  # Timesteps per update
parser.add_argument("--max_timesteps", default=1e6, type=int)
parser.add_argument("--K_epochs", default=80, type=int)  # PPO update iterations per timestep
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--render", action="store_true", help="Render the environment")
args = parser.parse_args()

# 创建环境
env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        action_mean = self.actor(state)
        return action_mean * self.max_action

    def act(self, state):
        action_mean = self.forward(state)
        action_distribution = Normal(action_mean, torch.ones_like(action_mean).to(device))
        action = action_distribution.sample()
        action_logprob = action_distribution.log_prob(action).sum(dim=-1, keepdim=True)
        return action, action_logprob


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.critic(state)


class PPO:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim).to(device)
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': args.lr},
            {'params': self.critic.parameters(), 'lr': args.lr}
        ])
        self.memory = []
        self.writer = SummaryWriter()

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action, action_logprob = self.actor.act(state)
        return action.cpu().detach().numpy(), action_logprob.cpu().detach().numpy()

    def store_transition(self, transition):
        self.memory.append(transition)

    def update(self):
        states, actions, rewards, next_states, logprobs, dones = zip(*self.memory)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        logprobs = torch.FloatTensor(logprobs).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # 计算优势估计
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            td_target = rewards + args.gamma * next_values * (1 - dones)
            advantages = td_target - values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        old_logprobs = logprobs

# PPO 更新
        for _ in range(args.K_epochs):
            # 重新计算动作概率和价值函数
            new_logprobs = self.actor.act(states)[1]
            state_values = self.critic(states).squeeze()

            # 计算概率比率
            ratios = torch.exp(new_logprobs - old_logprobs)

            # 计算策略损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - args.clip_eps, 1 + args.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 计算价值函数损失
            critic_loss = nn.MSELoss()(state_values, td_target)

            # 总损失
            loss = actor_loss + 0.5 * critic_loss

            # 梯度下降更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 清空经验池
        self.memory = []

    def save(self, checkpoint_path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


def main():
    ppo = PPO(state_dim, action_dim, max_action)
    timestep = 0
    episode = 0

    while timestep < args.max_timesteps:
        state = env.reset()
        episode_reward = 0
        for t in range(1, 10000):  # 每集最大步长设为10000
            if args.render:
                env.render()

            # 选择动作
            action, action_logprob = ppo.select_action(state)

            # 执行动作
            next_state, reward, done, _ = env.step(action)
            timestep += 1
            episode_reward += reward

            # 存储经验
            ppo.store_transition((state, action, reward, next_state, action_logprob, done))

            # 更新状态
            state = next_state

            # 如果达到更新频率，则进行网络更新
            if timestep % args.update_timestep == 0:
                ppo.update()
            if done:
                break

        print(f"Episode {episode}, Reward: {episode_reward}")
        episode += 1

        # 保存模型
        if episode % 100 == 0:
            ppo.save(f"ppo_checkpoint_{episode}.pth")

        # 如果时间步数达到最大，则结束训练
        if timestep >= args.max_timesteps:
            break

    env.close()


if __name__ == '__main__':
    main()