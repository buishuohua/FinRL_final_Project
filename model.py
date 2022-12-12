import torch.nn as nn
import numpy as np
import pandas as pd
import sys, os

from stable_baselines3.common.vec_env import DummyVecEnv

from env import StockLearningEnv
import config

class NeurelNetwork(nn.Module):
    """神经网络"""
    def __init__(self) -> None:
        pass

    def forward(out):
        return out


class Model():
    """做股票交易的智能体"""
    def __init__(self) -> None:
        pass

    def get_action(self, env):
        pass

    def learn(self, env):
        pass


def get_env(self, 
            train_data: pd.DataFrame, 
            trade_data: pd.DataFrame) -> DummyVecEnv:
    """分别返回训练环境和交易环境"""
    e_train_gym = StockLearningEnv(df = train_data,
                                                random_start = True,
                                                **config.ENV_PARAMS)
    env_train, _ = e_train_gym.get_sb_env()

    e_trade_gym = StockLearningEnv(df = trade_data,
                                                random_start = False,
                                                **config.ENV_PARAMS)
    env_trade, _ = e_trade_gym.get_sb_env()

    return env_train, env_trade

def save_model(self, model) -> None:
    pass

def train(model, train_df):
    pass

def test(model, test_df):
    pass

