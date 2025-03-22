import numpy as np

class Loss:
    def __init__(self):
        pass

    def MSELoss(self, label, prediction):
        return np.mean((label - prediction) ** 2)

    def BCELoss(self, label, prediction):
        return -np.mean(label * np.log(prediction + 1e-8) + (1 - label) * np.log(1 - prediction + 1e-8))

    def CELoss(self, label, prediction):
        return -np.mean(label * np.log(prediction + 1e-8))
