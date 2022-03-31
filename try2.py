import requests
from pprint import pprint

req = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT")

pprint(req.content)