from get_data import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pprint import pprint
from LSTM_regressor import *
from AUTO_regressor import *
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from pprint import pprint


regressor1 = AUTO_regressor(
    currency_symbol="BTCUSDT",
    seq_len=20,
    regressor=CatBoostRegressor(random_state=123, silent=True),
    ohlcv_size="1h",
    columns=["high", "low"],
    regressor_train_size=1000,
    version="baseline_seqlen_20"
)

data1 = get_data(
    "2022-01-01",
    "2022-04-07",
    "BTCUSDT",
    "1h"
)

regressor1.fill_up_cache(
    df=data1
)
