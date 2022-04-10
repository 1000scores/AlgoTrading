

class ModelAdmit:
    
    def __init__(self):
        pass
    
    def admit(self, score):
        if score[0] > 0.6:
            return "BUY"
        elif score[1] > 0.6:
            return "SELL"
        else:
            return "HOLD"
