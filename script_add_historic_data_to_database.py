from sqlalchemy import create_engine
import json
import pandas as pd
from account import Account
from common import date_to_milli, get_connection_and_tickers_to_database
from tqdm import tqdm
from get_data import download_data_df

if __name__ == "__main__":

    connection, tickers = get_connection_and_tickers_to_database()

    with connection:
        print(connection)
        for ticker in ticekrs:
            result_df = download_data_df(
                date_to_milli("2019-01-01"),
                None,
                ticker,
                "1m"
            )

            result_df.to_sql(ticker, connection)
    