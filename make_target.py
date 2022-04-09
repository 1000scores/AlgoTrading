import os
from get_data import *
from AUTO_regressor import *
from tqdm import tqdm

def make_target(
    start_date: str,
    end_date: str,
    file_name: str
):
    if not os.path.isdir(f"target"):
        os.mkdir(f"target")
        
    tickers = ["BTCUSDT", "ETHUSDT", "NEARUSDT", "BNBUSDT"]
    targets_ohlcv_sizes = ["1h", "12h", "1d"]
    regressor_features_ohlcv_sizes = ["1h", "12h", "1d"]
    regressor_features_columns = ["high", "low"]
    
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
                
                cur_avg = (cur_row["low"] + cur_row["high"] + cur_row["opentime"] + cur_row["closetime"]) / 4.0
                nxt_avg = (prev_row["low"] + prev_row["high"] + prev_row["opentime"] + prev_row["closetime"]) / 4.0
                
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
                        regressor_train_size=1000,
                        version="baseline_seqlen_20"
                    )
                    '''print("===============================")
                    print(cur_ohlcv_size)
                    print("===============================")'''
                    df_helper = get_data(
                        "2021-03-01",
                        "2022-04-07",
                        ticker,
                        cur_ohlcv_size
                    )
                    
                    for cur_col in regressor_features_columns:
                        cur_fname = f"regressor_{cur_col}_{cur_ohlcv_size}"
                        tmp_dict[cur_fname] = cur_auto_regressor.predict_by_opentime(df_helper, cur_col, cur_opentime)
                
                tmp_df = pd.DataFrame(tmp_dict, index=i-1)
                res_df = res_df.append(tmp_df)

            path_for_save = f"target/{file_name}_{target_ohlcv_size}_{start_date}_{end_date}"
            res_df.to_csv(path_for_save)
        
    
if __name__ == "__main__":
    make_target(
        "2022-03-01",
        "2022-04-01",
        "tmp"
    )