from sqlalchemy import create_engine
import json
import pandas as pd
from account import Account
from common import date_to_milli
from tqdm import tqdm
from get_data import download_data_df


def get_min_opentime(connection, ticker):
    get_min_data_db_query = None
    with open("sql/get_min_opentime.sql", 'r') as f:
        get_min_data_db_query = f.read()
    
    get_min_data_db_query = get_min_data_db_query.replace("TICKER", ticker)

    return int(pd.read_sql_query(get_min_data_db_query, connection).iloc[0, 0])


def get_max_closetime(connection, ticker):
    get_man_data_db_query = None
    with open("sql/get_max_closetime.sql", 'r') as f:
        get_man_data_db_query = f.read()
    
    get_man_data_db_query = get_man_data_db_query.replace("TICKER", ticker)

    return int(pd.read_sql_query(get_man_data_db_query, connection).iloc[0, 0])


def update_data(connection, last_closetime, ticker):
    result_df = download_data_df(
        last_closetime,
        None,
        ticker,
        "1m"
    )
    print(result_df)
    result_df.to_sql(ticker, connection, if_exists="append")
    

if __name__ == "__main__":
    config_path = "secrets/database_config.json"
    config = None
    with open(config_path, "rb") as f:
        config = json.load(f)

    engine = create_engine(f'mysql://{config["user"]}:{config["password"]}@{config["host"]}:{config["port"]}/{config["database"]}')
    with engine.connect() as connection:
        print(connection)
        for ticker in config["tickers"]:
            last_closetime = get_max_closetime(connection, ticker)
            print(last_closetime)
            update_data(connection, last_closetime, ticker)
