from get_data import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pprint import pprint
from LSTM_regressor import *
from AUTO_regressor import *
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from pprint import pprint
from tqdm import tqdm


def fill(
    start_date,
    end_date,
    currency_symbol,
    ohlcv_size,
    regressor_train_size
):
    regressor = AUTO_regressor(
        currency_symbol=currency_symbol,
        seq_len=20,
        regressor=CatBoostRegressor(random_state=123, silent=True),
        ohlcv_size=ohlcv_size,
        columns=["high", "low"],
        regressor_train_size=regressor_train_size,
        version="baseline_seqlen_20"
    )

    data = get_data(
        start_date,
        end_date,
        currency_symbol,
        ohlcv_size
    )

    regressor.fill_up_cache(
        df=data
    )

if __name__ == "__main__":
    currency_symbols = ["BTCUSDT"]#, "ETHUSDT", "NEARUSDT", "BNBUSDT"]
    ohlcv_sizes = ["1h", "12h", "1d"]
    start_date = "2019-01-01"
    end_date = "2022-04-07"
    
    ohlcv_to_train_size = {
        "1h" : 1000,
        "12h": 600,
        "1d": 360   
    }
    for currency_symbol in tqdm(currency_symbols):
        for ohlcv_size in tqdm(ohlcv_sizes):
            print(f"Filling {currency_symbol} size {ohlcv_size}")
            fill(
                start_date=start_date,
                end_date=end_date,
                currency_symbol=currency_symbol,
                ohlcv_size=ohlcv_size,
                regressor_train_size=ohlcv_to_train_size[ohlcv_size]
            )
            print()
