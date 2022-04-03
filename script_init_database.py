from sqlalchemy import create_engine
import json

if __name__ == "__main__":

    config_path = "secrets/database_config.json"
    config = None
    with open(config_path, "rb") as f:
        config = json.load(f)

    engine = create_engine(f'mysql://{config["user"]}:{config["password"]}@{config["host"]}:{config["port"]}/{config["database"]}')
    with engine.connect() as connection:
        print(connection)
        for ticker in config["tickers"]:
            create_db_query = None
            with open("sql/create_table.sql", 'r') as f:
                create_db_query = f.read()
            
            create_db_query = create_db_query.replace("TICKER_TABLE_NAME", ticker)
            
            print(create_db_query)

            connection.execute(create_db_query)