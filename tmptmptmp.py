
from get_data import *
import pandas as pd
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import StandardScaler
from pickle import load, dump

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
import math
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import torch
import torch.nn

# %%
'''
download_data(
    start_date="2021-12-01",
    end_date="2022-02-28",
    currency_symbol="BTCUSDT",
    path="data/train_15_2021-12-01_2022-02-28",
    ohlcv_size='15m'
)

download_data(
    start_date="2022-03-01",
    end_date="2022-03-30",
    currency_symbol="BTCUSDT",
    path="data/test_15_2022-03-01_2022-03-30",
    ohlcv_size='15m'
)


'''



# %%
PATH_TRAIN = "data/train_15_2021-12-01_2022-02-28"
PATH_TEST = "data/test_15_2022-03-01_2022-03-30"

SEQ_LEN = 35

# %%



def prepare_data(path, seq_len, scaler):

    tmp_X = get_high_from_data(path).reshape(-1, 1)
    tmp_X = scaler.transform(tmp_X).reshape(-1)
    X = []
    y = []

    for i in range(SEQ_LEN, len(tmp_X)):
        X.append(tmp_X[i - SEQ_LEN : i])
        y.append(tmp_X[i])
    
    return X, y

# %%
scaler = load(open('scalers/scaler_15m.pkl', 'rb'))

X_train, y_train = prepare_data(PATH_TRAIN, SEQ_LEN, scaler)
X_test, y_test = prepare_data(PATH_TEST, SEQ_LEN, scaler)

# %%
X_train = np.array(X_train)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
y_train = np.array(y_train)
X_test = np.array(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
y_test = np.array(y_test)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# %%
gpus = tf.config.experimental.list_physical_devices('GPU')
print("GPUS:")
print(gpus)
tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

# The LSTM architecture
regressor = Sequential()
# First LSTM layer with Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))
# Second LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Third LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Fourth LSTM layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
# The output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='rmsprop',loss='mean_squared_error')

# %%
regressor.summary()

# %%
# Fitting to the training set
regressor.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=2
)


# %%



