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
        create_db_query = None
        with open("sql/create_table.sql", 'r') as f:
            create_db_query = f.read()

        print(create_db_query)

        connection.execute(create_db_query)
        
    '''with connect(
        host=config['host'],
        port=config["port"],
        user=config['user'],
        password=config['password'],
        database=config['database']
    ) as connection:

        print(connection)
        create_db_query = None
        with open("sql/create_table.sql", 'r') as f:
            create_db_query = f.read()

        print(create_db_query)

        with connection.cursor() as cursor:
            cursor.execute(create_db_query)'''