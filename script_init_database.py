from mysql.connector import connect, Error
import json

if __name__ == "__main__":

    config_path = "secrets/database_config.json"
    config = None
    with open(config_path, "rb") as f:
        config = json.load(f)

    try:
        with connect(
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
                cursor.execute(create_db_query)


    except Error as e:
        print(e)