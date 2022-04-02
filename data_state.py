from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from pprint import pprint
from datetime import datetime
import json
from tqdm import tqdm
from account import Account
from common import date_to_milli
import numpy as np
from pathlib import Path
import os

class State:

    DATA_PATH = 'state_data'

    ASSETS = {
        'BTCUSDT',
        'ETHUSDT'
    }

    def __init__(self):

        self.ohlcv_size_dict = {
            '1m': Account.KLINE_INTERVAL_1MINUTE,
            '15m': Account.KLINE_INTERVAL_15MINUTE,
            '1h': Account.KLINE_INTERVAL_1HOUR,
            '12h': Account.KLINE_INTERVAL_12HOUR,
            '1d': Account.KLINE_INTERVAL_1DAY,
        }

        self.account = Account()
        
        if not Path(State.DATA_PATH).is_dir():
            os.mkdir(State.DATA_PATH)
        
        for asset in State.ASSETS:
            cur_path = f"{State.DATA_PATH}/state_{asset}"
            if not os.path.exists(cur_path):
                self.__download_data(
                    start_date="2018-01-01",
                    end_date=None,
                    currency_symbol=asset,
                    path=cur_path,
                    ohlcv_size="1m"
                )
        




    def __download_data(
        self,
        start_date: str,  # "2021-11-01"
        end_date,  # "2022-01-28"
        currency_symbol: str,  # "BTCUSDT"
        path: str,  # 'ohlcv1minute/test_2021-11-01_2022-01-28.txt'
        ohlcv_size: str  #  1m / 15m / 1h / 1d
    ):  

        ohlcv_generator = self.account.get_historical_klines_generator(
            currency_symbol,
            self.ohlcv_size_dict[ohlcv_size],
            date_to_milli(start_date),
            end_date if end_date is None else date_to_milli(end_date)
        )
        data = []
        #              0          1       2      3       4        5           6                7                   8               9               10               11
        columns = ['opentime', 'open', 'high', 'low', 'close', 'volume', 'closetime', 'quote_asset_volume', 'num_of_trades', 'taker_by_base', 'taker_buy_quote', 'ignore']
        for elem in tqdm(ohlcv_generator):
            data.append(elem)
        print(len(data))

            
        with open(path, 'w') as filehandle:
            json.dump(data, filehandle)


    def __get_high_from_data(self, path):
        with open(path, 'r') as f:
            return np.array([float(elem[2]) for elem in json.load(f)])
        
    def __get_low_from_data(self, path):
        with open(path, 'r') as f:
            return np.array([float(elem[3]) for elem in json.load(f)])