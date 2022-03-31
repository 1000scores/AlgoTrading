from os import access
from account import Account
from binance import Client
from binance_profile import Profile

__api_key = None
__api_secret = None
with open("api_secrets/api_key.txt", "r") as f:
        __api_key = f.read()
with open("api_secrets/api_secret.txt", "r") as f:
    __api_secret = f.read()
    
client = Client(__api_key, __api_secret)

client = Profile()
client.make_buy_order("BTCUSDT", 0.001, 30000)

'''print(client.get_account()) '''
'''order = client.create_order(
    symbol='BTCUSDT',
    side=client.SIDE_BUY,
    type="LIMIT_MAKER",
    quantity=0.001,
    price=10000
)
'''