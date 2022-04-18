import pandas as pd
from catboost import CatBoostClassifier
from model_admit import ModelAdmit
import matplotlib.pyplot as plt
from binance_profile import Profile


class OnlineModel:

    def __init__(
        self, 
        stack_percentage,
        stop_percentage,
        currency1,
        currency2
    ):
        self.stack_percentage = stack_percentage
        self.stop_percentage = stop_percentage
        self.currency1 = currency1
        self.currency2 = currency2
        self.profile = Profile()


    def decision(self):
        
