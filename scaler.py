from get_data import *
import pandas as pd
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import MinMaxScaler
from pickle import load, dump


def train_scaler(
    start_date,
    end_date,
    currency_symbol,
):
    train_data = np.array(get_data(
        start_date=start_date,
        end_date=end_date,
        currency_symbol=currency_symbol,
        ohlcv_size="1d"
    )["high"]).reshape(-1, 1)
    
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    
    scaler_path = f'scalers/scaler_{currency_symbol}.pkl'
    
    print(f"Saving scaler to {scaler_path}")
    dump(scaler, open(scaler_path, 'wb'))
    
    
def transform(data, currency_symbol):
    scaler_path = f'scalers/scaler_{currency_symbol}.pkl'
    
    scaler = load(open(scaler_path, 'rb'))
    
    return scaler.transform(data)# * 1000


def inverse_transform(data, currency_symbol):
    scaler_path = f'scalers/scaler_{currency_symbol}.pkl'
    
    scaler = load(open(scaler_path, 'rb'))
    
    return scaler.inverse_transform(data)# * 1000
    
    
if __name__ == "__main__":
    train_scaler(
        "2019-01-01",
        "2022-04-03",
        "BTCUSDT"
    )
    
    
    
        