from get_data import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pprint import pprint

def test_regressor(
    start_date: str,
    end_date: str,
    currency_symbol: str,
    ohlcv_size: str,
    testing_columns: list,
    regressor_lag: int,  # num of ohlcv as input to regressor
    obj_f_regressor,  # obj.predict(input, col, *args, **kwargs) -> output
    *args,
    **kwargs
):
    df_test = get_data(
        start_date,
        end_date,
        currency_symbol,
        ohlcv_size
    )

    pred = {}
    real = {}
    metrics = pd.DataFrame(data=[], columns=["mean_squared_error", "mean_absolute_error"], index=testing_columns)

    for col in testing_columns:
        pred[col] = []
        real[col] = []
        metrics[col] = None

    for i in range(regressor_lag, len(df_test)):
        for col in testing_columns:
            real[col].append(df_test.iloc[i][col])
            pred[col].append(obj_f_regressor.predict(
                input=df_test.iloc[i-regressor_lag : i],
                column=col,
                *args,
                **kwargs
            ))
    
    for col in testing_columns:
        metrics[col] = {
            "mean_squared_error" : mean_squared_error(real[col], pred[col]),
            "mean_absolute_error" : mean_absolute_error(real[col], pred[col])
        }

    pprint(metrics)

    return metrics
