from locale import currency
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from pprint import pprint
from datetime import datetime
import json
from tqdm import tqdm
from account import Account
from common import *
import numpy as np
import pandas as pd
from pprint import pprint
import os


def download_data_df(
    start_date,  # milli or "2021-11-01"
    end_date,  # None or milli or "2022-01-28"
    currency_symbol: str,  # "BTCUSDT"
    ohlcv_size: str  #  1m / 15m / 1h / 1d
):  
    
    account = Account()

    if type(start_date) == str:
        start_date = date_to_milli(start_date)

    if end_date is not None and type(end_date) == str:
        end_date = date_to_milli(end_date) - 1

    ohlcv_generator = account.get_historical_klines_generator(
        currency_symbol,
        get_ohlcv_kline_size_dict()[ohlcv_size],
        start_date,
        end_date
    )
    data = []
    #              0          1       2      3       4        5           6                7                   8               9               10               11
    columns = ['opentime', 'open', 'high', 'low', 'close', 'volume', 'closetime', 'quote_asset_volume', 'num_of_trades', 'taker_by_base', 'taker_buy_quote', 'ignore']
    for elem in tqdm(ohlcv_generator):
        data.append(elem)
    
    return pd.DataFrame(data, columns=columns)


def check_for_cache(
    start_date,
    end_date,
    currency_symbol,
    ohlcv_size
):
    
    if not os.path.isdir(f"train_data_cache"):
        os.mkdir(f"train_data_cache")
    
    fname = f"train_data_cache/{currency_symbol}_{ohlcv_size}_{start_date}_{end_date}.csv"
    if os.path.isfile(fname):
        return pd.read_csv(fname)
    
    return None

def get_data(
    start_date: str,
    end_date: str,
    currency_symbol: str,
    ohlcv_size: str  # num of minutes in OHLCV
):
    cache = check_for_cache(
        start_date,
        end_date,
        currency_symbol,
        ohlcv_size
    )
    if cache is not None:
        return cache
        
    get_interval_data_db_query = None
    with open("sql/get_date_interval.sql", 'r') as f:
        get_interval_data_db_query = f.read()
    

    get_interval_data_db_query = get_interval_data_db_query.replace("TICKER", currency_symbol)
    get_interval_data_db_query = get_interval_data_db_query.replace("START_MILLI", str(date_to_milli(start_date)))
    get_interval_data_db_query = get_interval_data_db_query.replace("END_MILLI", str(date_to_milli(end_date)))

    connection, _ = get_connection_and_tickers_to_database()

    ohlcv_minutes = int(get_ohlcv_to_minutes()[ohlcv_size])

    df_input = pd.read_sql_query(get_interval_data_db_query, connection)
    need_size = int((len(df_input) / ohlcv_minutes) * ohlcv_minutes)
    df_input = df_input.iloc[:need_size]
    df_input = df_input.drop('index', axis=1)
    df_input = df_input.astype({
        'opentime': 'int64',
        'open': 'float',
        'high': 'float',
        'low': 'float',
        'close': 'float',
        'volume': 'float',
        'closetime': 'int64',
        'quote_asset_volume': 'float',
        'num_of_trades': 'int64',
        'taker_by_base': 'float',
        'taker_buy_quote': 'float',
        'ignore': 'int64'
    })
    df_input = df_input.drop_duplicates(subset="opentime")
    df_input = df_input.sort_values(by=["opentime"])
    print(df_input)
    print("============================")
    minute_in_milli = 60000
    first_opentime = int(df_input.iloc[0]["opentime"])
    last_opentime = int(df_input.iloc[-1]["opentime"])
    lines_for_append = []
    last_line = None
    for cur_opentime in tqdm(range(first_opentime, last_opentime + minute_in_milli, minute_in_milli)):
        cur_row = df_input[df_input.opentime == cur_opentime]
        if len(cur_row) == 0:
            tmp = last_line.copy()
            tmp["opentime"] = cur_opentime
            lines_for_append.append(tmp)
        else:
            last_line = cur_row
    
    print(lines_for_append)
    
    for line in lines_for_append:
        df_input = df_input.append(line, ignore_index=True)
    
    df_input = df_input.sort_values(by=["opentime"]).reset_index(drop=True)
    print(df_input)


    df_res = pd.DataFrame([], columns=df_input.columns)

    print("Preparing data from database")
    for i in tqdm(range(0, len(df_input), ohlcv_minutes)):
        cur_ohlcv_df = df_input.iloc[i : i + ohlcv_minutes]
        
        cur_res_df = pd.DataFrame({
            'opentime': cur_ohlcv_df['opentime'].min(),
            'open': cur_ohlcv_df['open'].iloc[0],
            'high': cur_ohlcv_df['high'].max(),
            'low': cur_ohlcv_df['low'].min(),
            'close': cur_ohlcv_df['close'].iloc[len(cur_ohlcv_df['close']) - 1],
            'volume': cur_ohlcv_df['volume'].sum(),
            'closetime': cur_ohlcv_df['closetime'].max(),
            'quote_asset_volume': cur_ohlcv_df['quote_asset_volume'].sum(),
            'num_of_trades': cur_ohlcv_df['num_of_trades'].sum(),
            'taker_by_base': cur_ohlcv_df['taker_by_base'].sum(),
            'taker_buy_quote': cur_ohlcv_df['taker_buy_quote'].sum(),
            'ignore': cur_ohlcv_df['ignore'].min()  # =))
        }, index=[i // ohlcv_minutes])

        df_res = df_res.append(cur_res_df)
        
    fname = f"train_data_cache/{currency_symbol}_{ohlcv_size}_{start_date}_{end_date}.csv"
    df_res.to_csv(fname)
    return pd.read_csv(fname)


def get_high_from_data(path):
    with open(path, 'r') as f:
        return np.array([float(elem[2]) for elem in json.load(f)])
    

def get_low_from_data(path):
    with open(path, 'r') as f:
        return np.array([float(elem[3]) for elem in json.load(f)])

if __name__ == "__main__":
    print("here")
    df_mine = get_data("2019-01-01", "2022-04-08", "BTCUSDT", "1h")
    #df_true = download_data_df("2022-03-03", "2022-03-04", "ETHUSDT", "1h")
    #pprint(df_mine)
    #print()
    #pprint(df_true)