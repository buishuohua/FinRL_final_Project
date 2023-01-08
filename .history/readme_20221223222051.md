# 1. 数据

数据:

pulldata: 爬取数据

feature_engineering: 加技术指标

full_table: 对未上市数据置0

# 2. 环境

市场环境：

state_space = (1 + 资产数量 + 资产数量 * 资产指标数)

资产指标数 = ["open", "close", "high", "low", "volume"]

---

action_ space = (资产数量)  ， 区间为[-1, 1]

**随机开始**：

从训练期的前半段任意其中一天开始

**log记录**:

log_header: 每个信息的意思

log_step: 账户资产， 回撤相关的信息

**Reward计算：**

reward = 收益率 - 回撤率

收益率 = asset / initial_money

回撤率 = asset / max_asset -1

action的shape是什么样的？

# 3. Agent

基于各种算法对应的交易策略：

DDPG，A2C，PPO，TD3，SAC

我目前做了PPO的50000步，我下一步想做下ensemble的尝试，把5个模型的prediction预测综合下。
