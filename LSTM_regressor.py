from get_data import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler
from pickle import load, dump
import math
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from dataset import *
import matplotlib.pyplot as plt
import os


class RegressorClass(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.1):
        super(RegressorClass, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout_rate, num_layers=num_layers, batch_first=True)

        self.l1 = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()
        output, (h_out, _) = self.lstm1(inputs, (h0, c0))

        #print(f"forward1 {h_out.shape}")
        #print(f"hout afet view {h_out.shape}")
        h_out = self.l1(h_out[0]).flatten()
        
        #print(f"forward2 {h_out.shape}")

        return h_out


class LSTM_regressor:

    def __init__(
        self,
        train_start_date: str,
        train_end_date: str,
        test_start_date: str,
        test_end_date: str,
        currency_symbol: str,
        ohlcv_size: str,
        seq_len: int,
        columns: list,
        batch_size: int,
        hidden_size: int,
        dropout_rate: float,
        epochs: int,
        model_name: str
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.model_name = model_name

        self.train_dataloaders = {}
        self.test_dataloaders = {}

        for col in columns:
            tmp_train_dataset = AlgoTradingDataset(
                start_date=train_start_date,
                end_date=train_end_date,
                currency_symbol=currency_symbol,
                ohlcv_size=ohlcv_size,
                seq_len=self.seq_len,
                column=col
            )
            
            tmp_test_dataset = AlgoTradingDataset(
                start_date=test_start_date,
                end_date=test_end_date,
                currency_symbol=currency_symbol,
                ohlcv_size=ohlcv_size,
                seq_len=self.seq_len,
                column=col
            )

            self.train_dataloaders[col] = DataLoader(
                tmp_train_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
            
            self.test_dataloaders[col] = DataLoader(
                tmp_test_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        
        for col in columns:
            self.train(col)

    def get_model_path(self, col):
        return f"saved_models/{col}/{self.model_name}.pth"

    def train(
        self,
        col,
        log_loss_every=200
    ):

        model = RegressorClass(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=1,
            dropout_rate=self.dropout_rate,
        )
        
        

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        running_loss = 0.0
        loss_log = []
        for epoch in tqdm(range(self.epochs)):
            model.train()
            for i, data in enumerate(self.train_dataloaders[col]):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                outputs = model(inputs)
            
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                loss_log.append(loss.item())
                if i % log_loss_every == log_loss_every - 1:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / float(log_loss_every):.6f}')
                    running_loss = 0.0
            
            self.test(model, col)

        print('Finished Training')
        
        plt.plot(list(range(len(loss_log))), loss_log)
        plt.show()

        if not os.path.isdir(f"saved_models/{col}"):
            os.mkdir(f"saved_models/{col}")
        
        print(f"Saving model to {self.get_model_path(col)}")
        torch.save(model.state_dict(), self.get_model_path(col))

    
    def test(self, model, col):
        
        num_batches = len(self.test_dataloaders[col])
        
        criterion = nn.MSELoss()
        total_loss = 0
        model.eval()
        
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloaders[col]):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                outputs = model(inputs)
                total_loss += criterion(outputs, labels).item()
        
        avg_loss = total_loss / num_batches
        print(f"Test loss: {avg_loss}")
        
        return avg_loss

if __name__ == '__main__':
    print("here")
    reg = LSTM_regressor (
        train_start_date="2020-01-01",
        train_end_date="2021-05-01",
        test_start_date="2021-06-01",
        test_end_date="2022-04-01",
        currency_symbol="BTCUSDT",
        ohlcv_size="15m",
        seq_len=20,
        columns=["low", "high"],
        batch_size=16,
        hidden_size=3,
        dropout_rate=0,
        epochs=10,
        model_name="v1"
    )

    