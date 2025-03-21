import numpy as np

class Loss:
    def __init__(self, label, prediction):
        self.label = label
        self.prediction = prediction

    def MSEloss(self):
        return np.mean((self.label - self.prediction) ** 2)
