from datetime import datetime
import json
from sqlalchemy import create_engine
from account import Account


def date_to_milli(str_date):
    return int(datetime.strptime(str_date, "%Y-%m-%d").timestamp() * 1000)


def get_connection_and_tickers_to_database():
    config_path = "secrets/database_config.json"
    config = None
    with open(config_path, "rb") as f:
        config = json.load(f)

    engine = create_engine(f'mysql://{config["user"]}:{config["password"]}@{config["host"]}:{config["port"]}/{config["database"]}')

    return engine.connect(), config["tickers"]


def get_ohlcv_kline_size_dict():
    return {
        '1m': Account.KLINE_INTERVAL_1MINUTE,
        '15m': Account.KLINE_INTERVAL_15MINUTE,
        '1h': Account.KLINE_INTERVAL_1HOUR,
        '12h': Account.KLINE_INTERVAL_12HOUR,
        '1d': Account.KLINE_INTERVAL_1DAY,
    }
    
def get_ohlcv_to_minutes():
    return {
        '1m': 1,
        '15m': 15,
        '1h': 60,
        '12h': 720,
        '1d': 1440,
    }