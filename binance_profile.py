from account import Account


class Profile():
    
    def __init__(
        self,
        base_asset="USDT",
    ):
        self.client = Account()
        self.base_asset = base_asset
        
        
    def get_balance(self, asset):
        info = self.client.get_account()
        for elem in info["balances"]:
            if elem['asset'] == asset:
                return float(elem['free'])
    

    def make_buy_order(
        self,
        symbol: str,
        quantity: float,
        price: float,
    ):
        print(f"BUYING {symbol} FOR {quantity * price}")
        order = self.client.create_order(
            symbol=symbol,
            side=self.client.SIDE_BUY,
            type="LIMIT_MAKER",
            quantity=quantity,
            price=price
        )
        
    def make_sell_order(
        self,
        symbol: str,
        quantity: float,
        price: float,
    ):
        print(f"SELLING {symbol} FOR {quantity * price}")
        order = self.client.create_order(
            symbol=symbol,
            side=self.client.SIDE_SELL,
            type="LIMIT_MAKER",
            quantity=quantity,
            price=price
        )
    
    
    def make_symbol(self, buying_asset, selling_asset):
        return buying_asset + selling_asset
    