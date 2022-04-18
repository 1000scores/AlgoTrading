import os
from get_data import *
from AUTO_regressor import *
from tqdm import tqdm

def make_target(
    start_date: str,
    end_date: str,
    tickers: list,
    file_name: str
):
    if not os.path.isdir(f"target"):
        os.mkdir(f"target")
        
    targets_ohlcv_sizes = ["1h", "12h", "1d"]
    regressor_features_ohlcv_sizes = ["1h", "12h", "1d"]
    regressor_features_columns = ["high", "low"]
    
    last_ohlcv_columns_to_feature = ["open", "close", "high", "low", "volume", "num_of_trades"]
    
    ohlcv_to_train_size = {
        "1h" : 1000,
        "12h": 1000,
        "1d": 720
    }
    
    for ticker in tqdm(tickers):
        for target_ohlcv_size in tqdm(targets_ohlcv_sizes):
            train_df = get_data(
                start_date,
                end_date,
                ticker,
                target_ohlcv_size
            )
            res_columns = ["target"]
            for cur_ohlcv_size in regressor_features_ohlcv_sizes:
                for cur_col in regressor_features_columns:
                    res_columns.append(f"regressor_{cur_col}_{cur_ohlcv_size}")
            
            res_df = pd.DataFrame([], columns=res_columns)
            
            for i in tqdm(range(1, len(train_df))):
                cur_row = train_df.iloc[i]
                prev_row = train_df.iloc[i - 1]
                
                cur_avg = (cur_row["low"] + cur_row["high"] + cur_row["open"] + cur_row["close"]) / 4.0
                nxt_avg = (prev_row["low"] + prev_row["high"] + prev_row["open"] + prev_row["close"]) / 4.0
                label = int((nxt_avg - cur_avg) > 0)
                
                cur_opentime = train_df.iloc[i]["opentime"]
                
                tmp_dict = dict()
                tmp_dict["target"] = label
                
                for cur_ohlcv_size in regressor_features_ohlcv_sizes:
                    cur_auto_regressor = AUTO_regressor(
                        currency_symbol=ticker,
                        seq_len=20,
                        regressor=CatBoostRegressor(random_state=123, silent=True),
                        ohlcv_size=cur_ohlcv_size,
                        columns=regressor_features_columns,
                        regressor_train_size=ohlcv_to_train_size[cur_ohlcv_size],
                        version="baseline_seqlen_20"
                    )

                    df_helper = get_data(
                        "2019-01-01",
                        "2022-04-08",
                        ticker,
                        cur_ohlcv_size
                    )
                    
                    # Adding regressor features
                    for cur_col in regressor_features_columns:
                        cur_fname = f"regressor_{cur_col}_{cur_ohlcv_size}"
                        tmp_dict[cur_fname] = cur_auto_regressor.predict_by_opentime(df_helper, cur_col, cur_opentime)
                        
                    # Adding last ohlcv features
                    for last_ohlcv_col in last_ohlcv_columns_to_feature:
                        cur_fname = f"last_ohlcv_{last_ohlcv_col}_{cur_ohlcv_size}"
                        tmp_dict[cur_fname] = df_helper[df_helper.closetime < int(cur_opentime)].iloc[-1][last_ohlcv_col]
                
                res_df = res_df.append(pd.DataFrame(tmp_dict, index=[i-1]))

            path_for_save = f"target/{ticker}_{file_name}_{target_ohlcv_size}_{start_date}_{end_date}"
            res_df.to_csv(path_for_save)
        
    
if __name__ == "__main__":
    make_target(
        "2021-01-01",
        "2021-12-31",
        ["BTCUSDT"],
        "train"
    )
    
    make_target(
        "2022-01-01",
        "2022-04-08",
        ["BTCUSDT"],
        "test"
    )