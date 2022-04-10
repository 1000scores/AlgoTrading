import pandas as pd
from catboost import CatBoostClassifier
from model_admit import ModelAdmit
import matplotlib.pyplot as plt


def backtest_model_admit(
    currency_symbol,
    ohlcv_size
):
    
    df_test = pd.read_csv(f"target/{currency_symbol}_test_{ohlcv_size}_2022-01-01_2022-04-08").drop(["target"], axis=1)
    model = CatBoostClassifier()
    model.load_model(fname=f"prod_models/pred_{currency_symbol}_{ohlcv_size}")
    
    model_admit = ModelAdmit()
    
    money = 100.0
    crypto = 0.0
    fee_rate = 0.999
    
    in_money = True
    
    graph_X = []
    graph_y = []
    
    scores = []
    for i in range(0, len(df_test)):
        cur_row = df_test.iloc[i]
        cur_score = model.predict([cur_row], prediction_type="Probability")[0]
        scores.append(cur_score)
        decision = model_admit.admit(cur_score)
        
        cur_open = cur_row[f"last_ohlcv_open_{ohlcv_size}"]
        if decision == "BUY":
            if in_money:
                # print(f"Bought {currency_symbol} for {cur_open}")
                crypto += (money / cur_open) * fee_rate
                money = 0
                in_money = False
        elif decision == "SELL":
            if not in_money:
                # print(f"Sold {currency_symbol} for {cur_open}")
                money += (crypto * cur_open) * fee_rate
                crypto = 0
                in_money = True
        
        graph_X.append(i)
        graph_y.append(money + (crypto * cur_open) * fee_rate)

    plt.plot(graph_X, graph_y)
    plt.show()
    
    return graph_X, graph_y, scores


def backtest_model_class(
    currency_symbol,
    ohlcv_size
):
    
    df_test = pd.read_csv(f"target/{currency_symbol}_test_{ohlcv_size}_2022-01-01_2022-04-08").drop(["target"], axis=1)
    model = CatBoostClassifier()
    model.load_model(fname=f"prod_models/pred_{currency_symbol}_{ohlcv_size}")
    
    money = 100.0
    crypto = 0.0
    fee_rate = 0.999
    
    in_money = True
    
    graph_X = []
    graph_y = []
    
    for i in range(0, len(df_test)):
        cur_row = df_test.iloc[i]
        cur_class = model.predict([cur_row])[0]
        print(cur_class)
        decision = "BUY" if cur_class == 1 else "SELL"
        
        cur_open = cur_row[f"last_ohlcv_open_{ohlcv_size}"]
        if decision == "BUY":
            if in_money:
                # print(f"Bought {currency_symbol} for {cur_open}")
                crypto += (money / cur_open) * fee_rate
                money = 0
                in_money = False
        elif decision == "SELL":
            if not in_money:
                # print(f"Sold {currency_symbol} for {cur_open}")
                money += (crypto * cur_open) * fee_rate
                crypto = 0
                in_money = True
        
        graph_X.append(i)
        graph_y.append(money + (crypto * cur_open) * fee_rate)

    plt.plot(graph_X, graph_y)
    plt.show()
    
    return graph_X, graph_y