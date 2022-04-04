from get_data import *


def test_regressor(
    start_date: str,
    end_date: str,
    currency_symbol: str,
    ohlcv_size: str,
    regressor_lag: int,  # num of ohlcv us input to regressor
    wrapped_f_regressor,  # (input) -> output
    *args,
    **kwargs
):
    df_test = get_data(
        start_date,
        end_date,
        currency_symbol,
        ohlcv_size
    )


