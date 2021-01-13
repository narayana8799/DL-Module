import numpy as np

class KNN:

    def __init__(self, neighbours=11, p=2):
        self.K = neighbours
        self.P = p
        self.X, self.Y = None, None

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

    def predict(self, X):

        predictions = []
        for point in X:

            dist = np.power(self.X - point, self.P)
            dist = np.power(np.sum(dist, axis=1), 1/self.P)
            preds = np.argsort(dist)[:self.K]
            preds = list(self.Y[preds])
            unq_pts = sorted(np.unique(preds))
            predictions.append(np.argmax([preds.count(i) for i in unq_pts]))

        return predictions