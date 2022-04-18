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
from scaler import transform, inverse_transform


class RegressorClass(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, device):
        super(RegressorClass, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout_rate, num_layers=num_layers, batch_first=True)

        self.l1 = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().to(self.device)
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
        model_name: str,
        device
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.model_name = model_name
        self.currency_symbol = currency_symbol
        self.ohlcv_size = ohlcv_size
        self.device = device
        self.columns = columns
        
        self.train_dataloaders = {}
        self.test_dataloaders = {}
        
        self.models = dict()

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
        
    def fit(self):
        val_scores = []
        for col in self.columns:
            self.train(col)
            val_scores.append(self.test(col))
        
        return np.array(val_scores).mean()

    def get_model_path(self, col):
        return f"lstm_models/{self.currency_symbol}/{self.ohlcv_size}/{col}/{self.model_name}.pth"
    
    def save_trained_model(self, col):
        if not os.path.isdir(f"lstm_models"):
            os.mkdir(f"lstm_models")
            
        if not os.path.isdir(f"lstm_models/{self.currency_symbol}"):
            os.mkdir(f"lstm_models/{self.currency_symbol}")
        
        if not os.path.isdir(f"lstm_models/{self.currency_symbol}/{self.ohlcv_size}"):
            os.mkdir(f"lstm_models/{self.currency_symbol}/{self.ohlcv_size}")
            
        if not os.path.isdir(f"lstm_models/{self.currency_symbol}/{self.ohlcv_size}/{col}"):
            os.mkdir(f"lstm_models/{self.currency_symbol}/{self.ohlcv_size}/{col}")
        
        print(f"Saving model to {self.get_model_path(col)}")
        torch.save(self.models[col], self.get_model_path(col))
        
        
    def plot_loss(self, loss_log, col):
        if not os.path.isdir(f"lstm_loss_graphs/{self.currency_symbol}"):
            os.mkdir(f"lstm_loss_graphs/{self.currency_symbol}")
        
        if not os.path.isdir(f"lstm_loss_graphs/{self.currency_symbol}/{self.ohlcv_size}"):
            os.mkdir(f"lstm_loss_graphs/{self.currency_symbol}/{self.ohlcv_size}")
            
        if not os.path.isdir(f"lstm_loss_graphs/{self.currency_symbol}/{self.ohlcv_size}/{col}"):
            os.mkdir(f"lstm_loss_graphs/{self.currency_symbol}/{self.ohlcv_size}/{col}")
        plt.plot(list(range(len(loss_log))), loss_log)
        plt.savefig(f"lstm_loss_graphs/{self.currency_symbol}/{self.ohlcv_size}/{col}/{self.model_name}.png")
    

    def train(
        self,
        col,
        log_loss_every=500,
    ):

        self.models[col] = RegressorClass(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=1,
            dropout_rate=self.dropout_rate,
            device=self.device
        )
        
        self.models[col] = self.models[col].to(self.device)
        
        optimizer = torch.optim.Adam(self.models[col].parameters(), lr=0.001)
        criterion = nn.MSELoss()
        running_loss = 0.0
        loss_log = []
        for epoch in tqdm(range(self.epochs)):
            self.models[col].train()
            # print(f"Data loader len {len(self.train_dataloaders[col])}")
            for i, data in enumerate(self.train_dataloaders[col]):
                # get the inputs; data is a list of [inputs, labels]
                #print(i)
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.models[col](inputs)
            
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                loss_log.append(loss.item())
                if i % log_loss_every == log_loss_every - 1:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / float(log_loss_every):.6f}')
                    running_loss = 0.0
            
            self.test(col)

        print('Finished Training')
        
        self.plot_loss(loss_log, col)

        self.save_trained_model(col)

    
    def test(self, col):
        
        criterion = nn.MSELoss()
        total_loss = 0
        self.models[col].eval()
        
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloaders[col]):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.models[col](inputs)
                total_loss += criterion(outputs, labels).item()
        
        avg_loss = total_loss / float(len(self.test_dataloaders[col]))
        print(f"Test loss: {avg_loss}")
        
        return avg_loss


class TrainedLSTMRegressor:
    def __init__(self, model_name, currency_symbol, ohlcv_size, columns, device): 
        self.models = {}
        self.currency_symbol = currency_symbol
        self.device = device
        for col in columns:
            self.models[col] = torch.load(f"lstm_models/{currency_symbol}/{ohlcv_size}/{col}/{model_name}.pth")
            self.models[col].eval()
        
    def predict(self, inputs, column, opentime=None):
        with torch.no_grad():
            cur_model = self.models[column]
            cur_inputs = torch.Tensor(transform(np.array(inputs[column]).reshape((-1, 1)), self.currency_symbol)).to(self.device)
            cur_inputs = cur_inputs.reshape((1, -1, 1))
            return inverse_transform(cur_model(cur_inputs).cpu().reshape(-1, 1), self.currency_symbol).item()



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    '''columns = ["low", "high"]
    seq_lens = [25, 50, 75]
    batch_sizes = [8]
    hidden_sizes = [5, 15, 25]
    
    best_loss = 1e9
    
    best_hyper = None
    
    for seq_len in seq_lens:
        for batch_size in batch_sizes:
            for hidden_size in hidden_sizes:
                reg = LSTM_regressor (
                    train_start_date="2021-01-01",
                    train_end_date="2021-12-31",
                    test_start_date="2022-01-01",
                    test_end_date="2022-04-08",
                    currency_symbol="BTCUSDT",
                    ohlcv_size="1h",
                    seq_len=seq_len,
                    columns=["low", "high"],
                    batch_size=batch_size,
                    hidden_size=hidden_size,
                    dropout_rate=0,
                    epochs=70,
                    model_name="v3",
                    device=device
                )
                
                val = reg.fit()

                if val < best_loss:
                    best_loss = val
                    best_hyper = {
                        "seq_len" : seq_len,
                        "batch_size" : batch_size,
                        "hidden_size" : hidden_size
                    }
    print(best_loss)
    print(best_hyper)'''
    
    reg = LSTM_regressor (
        train_start_date="2021-01-01",
        train_end_date="2021-12-31",
        test_start_date="2022-01-01",
        test_end_date="2022-04-08",
        currency_symbol="BTCUSDT",
        ohlcv_size="1h",
        seq_len=50,
        columns=["low", "high"],
        batch_size=8,
        hidden_size=5,
        dropout_rate=0,
        epochs=70,
        model_name="v5",
        device=device
    )
    
    reg.fit()
    