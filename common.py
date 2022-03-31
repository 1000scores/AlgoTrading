from datetime import datetime

def date_to_milli(str_date):
    return int(datetime.strptime(str_date, "%Y-%m-%d").timestamp() * 1000)

