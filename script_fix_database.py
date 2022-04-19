
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
    for cur_opentime in tqdm(range(first_opentime, last_opentime + minute_in_milli, minute_in_milli)):
        cur_row = df_input[df_input.opentime == cur_opentime]
        if len(cur_row) == 0:
            tmp = last_line.copy()
            tmp["opentime"] = cur_opentime
            lines_for_append.append(tmp)
        else:
            last_line = cur_row
    
    for line in lines_for_append:
        df_input = df_input.append(line, ignore_index=True)
    
    df_input = df_input.sort_values(by=["opentime"]).reset_index(drop=True)

    df_input.to_sql(ticker, connection, if_exists="replace")

if __name__ == "__main__":
    connection, tickers = get_connection_and_tickers_to_database()