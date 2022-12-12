
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