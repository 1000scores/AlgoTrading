from get_data import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pprint import pprint
from LSTM_regressor import *
from AUTO_regressor import *
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from pprint import pprint

def test_regressor(
    start_date: str,
    end_date: str,
    currency_symbol: str,
    ohlcv_size: str,
    testing_columns: list,
    regressor_lag: int,  # num of ohlcv as input to regressor
    obj_f_regressor,  # obj.predict(input, col, *args, **kwargs) -> output
    silent
):
    df_test = get_data(
        start_date,
        end_date,
        currency_symbol,
        ohlcv_size
    )

    pred = {}
    real = {}
    metrics = {}

    for col in testing_columns:
        pred[col] = []
        real[col] = []

    log_every = 50
    for i in tqdm(range(regressor_lag, len(df_test))):
        for col in testing_columns:
            real[col].append(df_test.iloc[i][col])
            pred[col].append(obj_f_regressor.predict(
                inputs=df_test.iloc[i - regressor_lag : i],
                column=col,
            ).item())
            
        if not silent and i % log_every == 0:
            print(f"mse=>>>>{mean_squared_error(real[col], pred[col])}")
            print(f"mae=>>>>{mean_absolute_error(real[col], pred[col])}")
            print()
    
    
    
    
    for col in testing_columns:
        metrics[col] = {
            "mean_squared_error" : mean_squared_error(real[col], pred[col]),
            "mean_absolute_error" : mean_absolute_error(real[col], pred[col])
        }
    print()
    print("============================================")
    print("===============FINAL METRICS================")
    print("============================================")
    print()
    pprint(metrics)

    return metrics

if __name__ == "__main__":
    print("LSTM REGRESSOR:")
    test_regressor(
        start_date="2022-03-01",
        end_date="2022-04-05",
        currency_symbol="BTCUSDT",
        ohlcv_size="15m",
        testing_columns=["low", "high"],
        regressor_lag=20,
        obj_f_regressor=TrainedLSTMRegressor(
            model_name="v2",
            currency_symbol="BTCUSDT",
            ohlcv_size="15m",
            columns=["low", "high"],
            device=torch.device("cuda")
        ),
        silent=True
    )
    print()
    print()
    '''print("=====================================================")
    print()
    print("AUTO REGRESSOR")
    test_regressor(
        start_date="2022-01-01",
        end_date="2022-04-05",
        currency_symbol="BTCUSDT",
        ohlcv_size="1h",
        testing_columns=["low", "high"],
        regressor_lag=1000,
        obj_f_regressor=AUTO_regressor(
            currency_symbol="BTCUSDT",
            seq_len=10,
            regressor=CatBoostRegressor(random_state=123, silent=True)
        ),
        silent=False
    )'''