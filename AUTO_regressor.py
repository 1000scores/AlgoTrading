from skforecast.ForecasterAutoreg import ForecasterAutoreg
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from get_data import *
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from catboost import CatBoostRegressor
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import json


class AUTO_regressor:
    def __init__(
        self, 
        currency_symbol: str,
        seq_len: int,
        regressor,
        ohlcv_size: str,  # for cache
        columns: list,  # for cache
        regressor_train_size: int,  # for cache
        version: str,  # for cache
    ):

        self.currency_symbol = currency_symbol
        self.seq_len = seq_len
        self.regressor = regressor
        self.ohlcv_size = ohlcv_size
        self.columns = columns
        self.regressor_train_size = regressor_train_size
        self.version = version
        self.forecaster = ForecasterAutoreg(
            regressor=self.regressor,
            lags=self.seq_len
        )
        
        self.cache = dict()
       
        self.init_cache()


    
    def init_cache(self):
        if not os.path.isdir(f"regressor_cache"):
            os.mkdir(f"regressor_cache")
        
        if not os.path.isdir(f"regressor_cache/{self.currency_symbol}"):
            os.mkdir(f"regressor_cache/{self.currency_symbol}")
            
        if not os.path.isdir(f"regressor_cache/{self.currency_symbol}/{self.ohlcv_size}"):
            os.mkdir(f"regressor_cache/{self.currency_symbol}/{self.ohlcv_size}")

        if not os.path.isdir(f"regressor_cache/{self.currency_symbol}/{self.ohlcv_size}/ts_{self.regressor_train_size}"):
            os.mkdir(f"regressor_cache/{self.currency_symbol}/{self.ohlcv_size}/ts_{self.regressor_train_size}")
            
        self.cache_prefix = f"regressor_cache/{self.currency_symbol}/{self.ohlcv_size}/ts_{self.regressor_train_size}"
        
        for col in self.columns:
            if not os.path.isdir(f"{self.cache_prefix}/{col}"):
                os.mkdir(f"{self.cache_prefix}/{col}")
                
         
            if os.path.isfile(f"{self.cache_prefix}/{col}/{self.version}.json"):
                with open(f"{self.cache_prefix}/{col}/{self.version}.json", "r") as f:
                    self.cache[col] = json.load(f)
            else:
                self.cache[col] = dict()
    
    
    def predict_by_opentime(self, df, column, opentime: int):
        # print()
        # print(df[df.opentime >= opentime].head())
        # print()
        # print(str(int(opentime)))
        '''print(df.info())
        print()
        print(opentime)
        print()
        print(df.head(3))
        print()
        print(df.iloc[2307:])
        print()
        print(column)
        print()'''
        last_dfs = (df.opentime <= int(opentime))
        last_opentime = df[last_dfs].iloc[-1]["opentime"]
        index_predict = df.index[last_dfs].tolist()[-1]
        return self.predict(
            df.iloc[index_predict - self.regressor_train_size: index_predict],
            column,
            opentime=last_opentime
        )
    
    
    def predict(self, inputs, column, opentime=None):
        # assert len(inputs) == self.regressor_train_size
        if str(opentime) in self.cache[column]:
            #print("here")
            return self.cache[column][str(opentime)]
        '''print()
        print(f"path = {f'{self.cache_prefix}/{column}/{self.version}.json'}")
        print(f"opentime = {opentime}")
        print(f"column = {column}")
        #print(self.cache[column])
        print()'''
        self.forecaster.fit(y=pd.Series(inputs[column]))
        return self.forecaster.predict(steps=1)

    
    def fill_up_cache(self, df):
        print(len(df))
        for col in self.columns:
            for i in tqdm(range(self.regressor_train_size, len(df))):
                cur_name = str(df.iloc[i]["opentime"])
                
                if cur_name in self.cache[col]:
                    continue
                
                self.cache[col][cur_name] = self.predict(
                    inputs=df.iloc[i - self.regressor_train_size : i],
                    column=col,
                ).item()
                
            with open(f"{self.cache_prefix}/{col}/{self.version}.json", "w") as f:
                json.dump(self.cache[col], f)
    