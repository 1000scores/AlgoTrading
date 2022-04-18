

class ModelAdmit:
    
    def __init__(self, thresh1=0.6, thresh2=0.6):
        self.thresh1 = thresh1
        self.thresh2 = thresh2
    
    def admit(self, score):
        if score[0] > self.thresh1:
            return "BUY"
        elif score[1] > self.thresh2:
            return "SELL"
        else:
            return "HOLD"
