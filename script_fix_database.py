from sqlalchemy import create_engine
import json
import pandas as pd
from account import Account
from common import date_to_milli, get_connection_and_tickers_to_database
from tqdm import tqdm
from get_data import download_data_df


def fix_database(connection, ticker):

    get_table_query = None
    with open("sql/get_table.sql", 'r') as f:
        get_table_query = f.read()

    get_table_query = get_table_query.replace("TICKER", ticker)

    df_input = pd.read_sql_query(get_table_query, connection)

    df_input = df_input.drop('index', axis=1)
    df_input = df_input.astype({
        'opentime': 'int64',
        'open': 'float',
        'high': 'float',
        'low': 'float',
        'close': 'float',
        'volume': 'float',
        'closetime': 'int64',
        'quote_asset_volume': 'float',
        'num_of_trades': 'int64',
        'taker_by_base': 'float',
        'taker_buy_quote': 'float',
        'ignore': 'int64'
    })
    df_input = df_input.drop_duplicates(subset="opentime")
    df_input = df_input.sort_values(by=["opentime"])

    minute_in_milli = 60000
    first_opentime = int(df_input.iloc[0]["opentime"])
    last_opentime = int(df_input.iloc[-1]["opentime"])
    lines_for_append = []
    last_line = None

    cache_data = None
    cache_path = f"fix_cache/fix_{ticker}_cache.json"

    if not os.path.isdir("fix_cache"):
        os.mkdir("fix_cache")

    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            cache_data = json.load(f)
    else:
        cache_data = {"last_opentime_fix" : first_opentime}

    for cur_opentime in tqdm(range(cache_data["last_opentime_fix"], last_opentime + minute_in_milli, minute_in_milli)):
        cur_row = df_input[df_input.opentime == cur_opentime]
        if len(cur_row) == 0:
            tmp = last_line.copy()
            tmp["opentime"] = cur_opentime
            tmp["closetime"] = cur_opentime + minute_in_milli - 1
            tmp_df = pd.DataFrame(tmp, columns=df_input.columns)
            tmp_df.to_sql(ticker, connection, if_exists="append")
            print(f"appended = {cur_opentime}")
        else:
            last_line = cur_row
        
        cache_data["last_opentime_fix"] = cur_opentime

    with open(cache_path, "wb") as f:
        json.dump(cache_path, f)


if __name__ == "__main__":
    connection, tickers = get_connection_and_tickers_to_database()

    for ticker in tickers:
        fix_database(connection, ticker)