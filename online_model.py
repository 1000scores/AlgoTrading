import pandas as pd
from catboost import CatBoostClassifier
from model_admit import ModelAdmit
import matplotlib.pyplot as plt
from binance_profile import Profile
from get_data import *
from AUTO_regressor import *


class OnlineModel:

    def __init__(
        self,
        prediction_model,  # with .predict([cur_row], prediction_type="Probability")[0] func
        admit_model,
        stack_percentage: float,
        stop_percentage: float,
        currency1: str,
        currency2: str,
        ohlcv_size: str,
        regressor_train_size: int = 1000,
        regressor_seq_len: int = 20,
    ):
        self.prediction_model = prediction_model
        self.admit_model = admit_model
        self.stack_percentage = stack_percentage
        self.stop_percentage = stop_percentage
        self.currency1 = currency1
        self.currency2 = currency2
        self.ohlcv_size = ohlcv_size
        self.ticker = self.currency1 + self.currency2
        self.regressor_train_size = regressor_train_size
        self.regressor_seq_len = regressor_seq_len
        self.profile = Profile()

        self.regressor_features_ohlcv_sizes = ["1h", "12h", "1d"]
        self.regressor_features_columns = ["high", "low"]
        self.last_ohlcv_columns_to_feature = ["open", "close", "high", "low", "volume", "num_of_trades"]

        self.auto_regressor = AUTO_regressor(
            currency_symbol=self.ticker,
            seq_len=self.regressor_seq_len,
            regressor=CatBoostRegressor(random_state=123, silent=True),
            ohlcv_size=self.ohlcv_size,
            columns=self.regressor_features_columns,
            regressor_train_size=1000,
            version="baseline_seqlen_20"
        )


    def decision(self):
        
        tmp_dict = dict()
        for cur_ohlcv_size in self.regressor_features_ohlcv_sizes:
            for cur_col in self.regressor_features_columns:
                cur_fname = f"regressor_{cur_col}_{cur_ohlcv_size}"
                tmp_dict[cur_fname] = self.auto_regressor.predict(
                    inputs=online_get_latest_data(self.regressor_train_size, cur_ohlcv_size, self.ticker),
                    column=cur_col
                )
                
            # Adding last ohlcv features
            for last_ohlcv_col in self.last_ohlcv_columns_to_feature:
                cur_fname = f"last_ohlcv_{last_ohlcv_col}_{cur_ohlcv_size}"
                tmp_dict[cur_fname] = online_get_latest_data(1, cur_ohlcv_size, self.ticker)
            
        score = self.prediction_model.predict(pd.DataFrame(tmp_dict, index=[0]).iloc[0])[0]

        return admit_model(score)

if __name__ == "__main__":
    prediction_model = CatBoostClassifier()
    prediction_model.load_model(fname=f"prod_models/pred_{'BTCUSDT'}_{'1h'}")
    online_model = OnlineModel(
        prediction_model=prediction_model,
        admit_model=ModelAdmit(thresh1=0.5, thresh2=0.5),
        stack_percentage=0.25,
        stop_percentage=1,
        currency1="BTC",
        currency2="USDT",
        ohlcv_size="1h"
    )

    print(online_model.decision())