import numpy as np
from brain.utils.constants import LR, EPOCHS
class LinearRegression:

    def __init__(self, learning_rate=LR, epochs=EPOCHS):

        self.lr = learning_rate
        self.epochs = epochs
        self.X, self.Y = None, None
        self.dW, self.dB = None, None
        self.Z, self.info = None, None

    def fit(self, X, Y):
        self.X = X.T
        self.Y = Y
        self.optimize()

    def predict(self, X):
        pass

    def optimize(self):
        pass
    
