from sqlalchemy import create_engine
import json
import pandas as pd
from account import Account
from common import date_to_milli
from tqdm import tqdm
from get_data import download_data_df

if __name__ == "__main__":
    config_path = "secrets/database_config.json"
    config = None
    with open(config_path, "rb") as f:
        config = json.load(f)

    engine = create_engine(f'mysql://{config["user"]}:{config["password"]}@{config["host"]}:{config["port"]}/{config["database"]}')
    with engine.connect() as connection:
        print(connection)
        for ticker in config["tickers"]:
            
            result_df = download_data_df(
                date_to_milli("2019-01-01"),
                None,
                ticker,
                "1m"
            )

            result_df.to_sql(ticker, connection)
    