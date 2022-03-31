from binance import Client


class Account(Client):
    def __init__(self) -> None:
        self.__api_key = None
        self.__api_secret = None
        self.client = None
        
        with open("api_secrets/api_key.txt", "r") as f:
            self.__api_key = f.read()
        with open("api_secrets/api_secret.txt", "r") as f:
            self.__api_secret = f.read()
        
        Client.__init__(self, self.__api_key, self.__api_secret)
    