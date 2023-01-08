
Start_Trade_Date = "2009-01-01"
End_Trade_Date = "2019-01-01"
End_Test_Date = "2021-01-01"

# 技术指标列表
TECHNICAL_INDICATORS_LIST = [
    "boll_ub", "boll_lb", "rsi_20", "close_20_sma", "close_60_sma", "close_120_sma", \
    "macd", "volume_20_sma", "volume_60_sma", "volume_120_sma"
]



# 环境的超参数
information_cols = TECHNICAL_INDICATORS_LIST + ["close", "day", "amount", "change", "daily_variance"]
ENV_TRAIN_PARAMS = {
    "initial_amount": 1e6,
    "hmax": 5000, 
    "currency": '￥',
    "buy_cost_pct": 3e-3,
    "sell_cost_pct": 3e-3,
    "cache_indicator_data": True,
    "daily_information_cols": information_cols, 
    "print_verbosity": 500,
    "patient":True,
}
ENV_TRADE_PARAMS = {
    "initial_amount": 1e6,
    "hmax": 5000, 
    "currency": '￥',
    "buy_cost_pct": 3e-3,
    "sell_cost_pct": 3e-3,
    "cache_indicator_data": True,
    "daily_information_cols": information_cols, 
    "print_verbosity": 500,
    "random_start": False,
    "patient":True
}

# 强化学习Stable_baselines3模型列表
MODEL_LIST = ["a2c", "ddpg", "ppo", "sac", "td3"]

# 模型的超参数
A2C_PARAMS = {
    "n_steps": 5, 
    "ent_coef": 0.01, 
    "learning_rate": 0.0007
    }
PPO_PARAMS = {
    "n_steps": 256,
    "ent_coef": 0.01,
    "learning_rate": 0.00005,
    "batch_size": 256
    }
DDPG_PARAMS = {
    "batch_size": 128, 
    "buffer_size": 50000, 
    "learning_rate": 0.001
    }
TD3_PARAMS = {
    "batch_size": 100, 
    "buffer_size": 1000000, 
    "learning_rate": 0.001
    }
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 2000,
    "ent_coef": "auto_0.1"
}

# 环境的超参数
information_cols = TECHNICAL_INDICATORS_LIST + ["close", "day", "amount", "change", "daily_variance"]
ENV_PARAMS = {
    "initial_amount": 1e6,
    "hmax": 5000, 
    "currency": '￥',
    "buy_cost_pct": 3e-3,
    "sell_cost_pct": 3e-3,
    "cache_indicator_data": True,
    "daily_information_cols": information_cols, 
    "print_verbosity": 500,
    "patient":True,
}

# tensorboard_log 路径
TENSORBOARD_LOG_DIR = f"tensorboard_log"