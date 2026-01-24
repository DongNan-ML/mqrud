import numpy as np

class CustomEarlyStopCallback():
    
    def __init__(self, key):
        self.loss_list = []
        self.key = key

    def after_iteration(self, info):
        loss_history = info.metrics['learn'][self.key]
        # At least train with 50 iterations:
        if len(loss_history) < 50:
            return True
            
        avg_loss_last = np.sum(loss_history[-5:]) / 5
        avg_loss_previous = np.sum(loss_history[-10:-5]) / 5
        loss_decrease = avg_loss_previous - avg_loss_last

        if loss_decrease <= 0.0001:
            return False
        else:
            return True