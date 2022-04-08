from get_data import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pprint import pprint
from LSTM_regressor import *
from AUTO_regressor import *
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from pprint import pprint






class DecisionCatboost:
    def __init__(
        self,
        train_start_date,
        train_end_date,
    ):
        pass