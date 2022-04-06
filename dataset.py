from get_data import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler
from pickle import load, dump
import math
from sklearn.metrics import mean_squared_error
from scaler import transform


class AlgoTradingDataset(Dataset):
    def __init__(self, start_date, end_date, currency_symbol, ohlcv_size, seq_len, column):
        df = get_data(
            start_date,
            end_date,
            currency_symbol,
            ohlcv_size
        )
        
        tmp_X = np.array(df[column]).reshape(-1, 1)
        tmp_X = transform(tmp_X, currency_symbol).reshape(-1)
        self.X = []
        self.y = []

        for i in range(seq_len, len(tmp_X)):
            self.X.append(tmp_X[i - seq_len : i])
            self.y.append(tmp_X[i])
        self.X = torch.Tensor(self.X).unsqueeze(dim=-1)
        self.y = torch.Tensor(self.y)


    def __len__(self):
        return self.X.shape[0]

    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

if __name__ == "__main__":
    training_data = AlgoTradingDataset(
        start_date="2022-01-01",
        end_date="2022-01-30",
        currency_symbol="BTCUSDT",
        ohlcv_size="15m",
        seq_len=40,
        column="low"
    )
    train_dataloader = DataLoader(training_data, batch_size=8, shuffle=False)
    print(next(iter(train_dataloader))[0].shape)
    print(next(iter(train_dataloader))[1].shape)
