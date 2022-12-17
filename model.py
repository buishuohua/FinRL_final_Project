import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.distributions import Categorical

import numpy as np
import pandas as pd
import sys, os
import matplotlib.pyplot as plt

import gym
from gym.spaces import Box
from stable_baselines3.common.vec_env import DummyVecEnv


from env import StockLearningEnv
import config
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
df_train = pd.read_csv("./data_file/train.csv")
df_train = df_train.iloc[:1000, :]
df_trade = pd.read_csv("./data_file/trade.csv")

def get_env_PPO(
        data: pd.DataFrame, 
        if_train) -> DummyVecEnv:
    if if_train:
        e_gym = StockLearningEnv(df = data,
                                            random_start = True,
                                            **config.ENV_PARAMS)
    else:
        e_gym = StockLearningEnv(df = data,
                                            random_start = False,
                                            **config.ENV_PARAMS)
    env, init_obs = e_gym.get_sb_env()

    return env, init_obs
class PPO_NN(nn.Module):
    def __init__(self, input_dim, output_dim): 
        super().__init__()
        self.actor = nn.Sequential(
            # nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
            # nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
            # nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim * 3)
        )
        self.critic = nn.Sequential(
            # nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
            # nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=1, padding=0),
            # nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        print(input_dim)
    def forward(self, x):
        actor_output = self.actor(x)
        critic_output = self.critic(x)
        actor_outout_matrix = actor_output.view(-1, 3)
        return Categorical(logits=actor_outout_matrix), critic_output

class PPO_Agent:
    def __init__(self, data = df_train, if_train = True):
        self.env, self.init_obs = get_env_PPO(data, if_train)
        self.number_of_iterations = 1601
        self.gamma = 0.95
        self.learning_rate_actor = 0.00025
        self.learning_rate_critic = 0.001

        self.batch_size = self.env.time_spread
        self.num_miniBatch = 4
        self.miniBatch_size = self.batch_size // self.num_miniBatch

        self.lamda = 0.95
        self.epochs = 30
        self.clip_value = 0.2

        self.save_directory = "./PPO"
        self.all_episode_rewards = []
        self.all_mean_rewards = []
        self.episode = 0

        input_dim = self.env.observation_space.shape[0]
        output_dim = self.env.action_space.shape[0]
        self.model = PPO_NN(input_dim, output_dim).to(device)
        self.model_old = PPO_NN(input_dim, output_dim).to(device)
        self.model_old.load_state_dict(self.model.state_dict()) 
        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam([
            {'params': self.model.actor.parameters(), 'lr': self.learning_rate_actor},
            {'params': self.model.critic.parameters(), 'lr': self.learning_rate_critic}
             ], eps=1e-4)

        self.rewards = []
        self.obs = self.init_obs.__array__()

    def save_checkpoint(self):
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        filename = os.path.join(self.save_directory, 'checkpoint_{}.pth'.format(self.episode))
        torch.save(self.model_old.state_dict(), f=filename)
        print('Checkpoint saved to \'{}\''.format(filename))

    # 读取模型
    def load_checkpoint(self, filename):
        self.model.load_state_dict(torch.load(os.path.join(self.save_directory, filename)))
        self.model_old.load_state_dict(torch.load(os.path.join(self.save_directory, filename)))
        print('Resuming training from \'{}\'.'.format(filename))
        return int(filename[11:-4]) # 返回第几轮

    def sample(self):
        # 初始化，都附上0值
        rewards = np.zeros(self.batch_size, dtype=np.float32) # 需要判断出开始点到结束点的长度
        actions = np.zeros((self.batch_size, len(self.env.assets)), dtype=np.float32)
        done = np.zeros(self.batch_size, dtype=bool)
        obs = np.zeros((self.batch_size, self.env.state_space), dtype=np.float32)
        log_pis = np.zeros((self.batch_size, len(self.env.assets)), dtype=np.float32)
        values = np.zeros(self.batch_size, dtype=np.float32)

        # 开始循环，每次采样env的随机开始点到terminal
        for t in range(self.batch_size):
            # 计算这些的时候不需要求导数，这个会让计算变得更快
            with torch.no_grad():
                # 初始化状态
                obs[t] = self.obs
                # print(t)
                # print("========================")
                # print(self.obs)
                # 动作种类与V
                pi, v = self.model_old(torch.tensor(self.obs, dtype=torch.float32, device=device).unsqueeze(0))
                values[t] = v.cpu().numpy()
                a = pi.sample()
                actions[t] = a.cpu().numpy()
                log_pis[t] = pi.log_prob(a).cpu().numpy()
                # print(a)
                # print("===========================")
                # print(pi.log_prob(a))
            self.obs, rewards[t], done[t], _ = self.env.step(actions[t])
            self.obs = np.array(self.obs)
            self.rewards.append(rewards[t])

            if done[t]:
                self.episode += 1
                self.all_episode_rewards.append(np.sum(self.rewards))
                self.rewards = []
                # 终结状态初始化环境
                self.env, self.obs = get_env_PPO(df_train, True)
                self.obs = self.obs.__array__()

                # 每10轮输出一下得到的平均收益
                if self.episode % 10 == 0:
                    mean_rewards = np.mean(self.all_episode_rewards[-10:])
                    print('Episode: {} - {}, average reward: {}'.format(self.episode-9, self.episode, mean_rewards))
                    self.all_mean_rewards.append(mean_rewards)
                # 每100轮画一下图并保存模型
                if self.episode % 100 == 0:
                    plt.plot(self.all_mean_rewards)
                    plt.xlabel('Batch Number (Epidodes / 10)')
                    plt.ylabel('Average Return')
                    plt.savefig("{}/mean_reward_{}.png".format(self.save_directory, self.episode))
                    plt.clf()
                    self.save_checkpoint()

        # 计算回报returns和优势函数advantages
        returns, advantages = self.calculate_advantages(done, rewards, values)
        # 把rewards, actions, done, obs, log_pis, values存到一个字典并返回
        samples = {
            'obs': torch.tensor(obs.reshape(obs.shape[0], *obs.shape[1:]), dtype=torch.float32, device=device),
            'actions': torch.tensor(actions, device=device),
            'values': torch.tensor(values, device=device),
            'log_pis': torch.tensor(log_pis, device=device),
            'advantages': torch.tensor(advantages, device=device, dtype=torch.float32),
            'returns': torch.tensor(returns, device=device, dtype=torch.float32)
        }
        return samples

    # 计算优势函数
    def calculate_advantages(self, done, rewards, values):
        _, last_value = self.model_old(torch.tensor(self.obs, dtype=torch.float32, device=device).unsqueeze(0))
        last_value = last_value.cpu().data.numpy()
        values = np.append(values, last_value)
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            mask = 1.0 - done[i]
            delta = rewards[i] + self.gamma * values[i + 1] * mask - values[i]
            gae = delta + self.gamma * self.lamda * mask * gae
            returns.insert(0, gae + values[i])
        adv = np.array(returns) - values[:-1]
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-8)

    # 使用收集的数据训练模型
    def train(self, samples):
        indexes = torch.randperm(self.batch_size) # 返回一个0 ~ 路径长度-1 之间的随机下标列表
        for start in range(0, self.batch_size, self.miniBatch_size):
            end = start + self.miniBatch_size
            mini_batch_indexes = indexes[start: end]
            mini_batch = {}
            for k, v in samples.items():
                mini_batch[k] = v[mini_batch_indexes]
            for _ in range(self.epochs):
                loss = self.calculate_loss(mini_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.model_old.load_state_dict(self.model.state_dict())

    # 计算损失函数 loss
    def calculate_loss(self, samples):
        sampled_returns = samples['returns']
        sampled_advantages = samples['advantages']
        pi, value = self.model(samples['obs'])
        losses = []
        for i in range(self.batch_size):
            ratio = torch.exp(pi.log_prob(samples['actions'][i]) - samples['log_pis'][i])
            clipped_ratio = ratio.clamp(min=1.0 - self.clip_value, max=1.0 + self.clip_value)
            policy_reward = torch.min(ratio * sampled_advantages, clipped_ratio * sampled_advantages)
            entropy_bonus = pi.entropy()
            vf_loss = self.mse_loss(value, sampled_returns)
            loss = -policy_reward + 0.5 * vf_loss - 0.01 * entropy_bonus
            losses.append(loss)
        return losses.mean()

def PPO_main(mode):
    PPO = PPO_Agent()
    if mode == "train":
        while PPO.episode <= PPO.number_of_iterations:
            samples = PPO.sample()
            PPO.train(samples)
    elif mode == "test":
        pass

PPO_main("train")
