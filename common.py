from datetime import datetime
import json
from sqlalchemy import create_engine


def date_to_milli(str_date):
    return int(datetime.strptime(str_date, "%Y-%m-%d").timestamp() * 1000)


def get_connection_and_tickers_to_database():
    config_path = "secrets/database_config.json"
    config = None
    with open(config_path, "rb") as f:
        config = json.load(f)

    engine = create_engine(f'mysql://{config["user"]}:{config["password"]}@{config["host"]}:{config["port"]}/{config["database"]}')

    return engine.connect(), config["tickers"]
