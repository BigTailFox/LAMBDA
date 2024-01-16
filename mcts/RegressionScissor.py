#!usr/bin/python
# coding=utf8
import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class RegressorBase(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.error = np.nan

    @abstractmethod
    def train(self, X, Y, weight=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class SGD(RegressorBase):
    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._sgd = SGDRegressor(
            loss="squared_loss", penalty="l1", alpha=1e-4, max_iter=50000, tol=1e-5
        )
        self.sgd = make_pipeline(self._scaler, self._sgd)

    def train(self, X, Y, weight=None) -> None:
        if weight is not None:
            weight = weight * len(weight)
        self.sgd.fit(X, Y, sgdregressor__sample_weight=weight)
        self.error = self.sgd.score(X, Y, sample_weight=weight)

    def predict(self, X) -> np.ndarray:
        return self.sgd.predict(X)


class RegressionScissor(object):
    def __init__(self) -> None:
        super().__init__()
        self.regressor = SGD()
        self.good_mean = np.nan
        self.bad_mean = np.nan
        self.good_label = 1
        self.bad_label = 0
        self.labels = np.array([])
        self.error = np.nan
        self.mean = np.nan
        self.debug = False

    def get_mean(self, Y, W):
        return Y.mean() if W is None else np.sum(Y * W)

    def train(self, X, Y, W):
        self.mean = self.get_mean(Y, W)
        self.regressor.train(X, Y, W)

    def predict(self, X, returnY=False):
        Y = self.regressor.predict(X)
        L = (Y >= self.mean).astype("int")
        if returnY:
            return L, Y
        return L

    def try_split(self, samples, ids, weights=None, min_lift=0, returnY=False):
        X = samples[:, :-1]
        Y = samples[:, -1]
        W = weights
        self.train(X, Y, W)
        self.error = self.regressor.error
        if returnY:
            self.labels, Yhat = self.predict(X, returnY=True)
        else:
            self.labels = self.predict(X)
        if len(np.unique(self.labels)) < 2:
            if self.debug:
                print(
                    "[DEBUG] RegressionScissor.try_split() failed, all points with one label"
                )
            if returnY:
                return False, None, None, Yhat
            else:
                return False, None, None
        maskA = self.labels == self.good_label
        maskB = self.labels == self.bad_label
        if W is not None:
            self.good_mean = np.sum(Y[maskA] * W[maskA]) / np.sum(W[maskA])
            self.bad_mean = np.sum(Y[maskB] * W[maskB]) / np.sum(W[maskB])
        else:
            self.good_mean = np.mean(Y[maskA])
            self.bad_mean = np.mean(Y[maskB])
        if self.good_mean < self.bad_mean:
            self.good_mean, self.bad_mean = self.bad_mean, self.good_mean
            self.good_label, self.bad_label = self.bad_label, self.good_label
        if self.good_mean - self.bad_mean < self.mean * min_lift:
            if self.debug:
                print("[DEBUG] RegressionScissor:try_split() failed, for unenough lift")
                print(
                    "        with density:  good mean: {:.4f}, bad mean {:.4f}, origin mean: {:.4f}".format(
                        self.good_mean, self.bad_mean, self.mean
                    )
                )
                print(
                    "        w/o  density:  good mean: {:.4f}, bad mean {:.4f}, origin mean: {:.4f}".format(
                        np.mean(Y[maskA]), np.mean(Y[maskB]), np.mean(Y)
                    )
                )
            if returnY:
                return False, None, None, Yhat
            else:
                return False, None, None
        if returnY:
            return (
                True,
                ids[self.labels == self.good_label],
                ids[self.labels == self.bad_label],
                Yhat,
            )
        else:
            return (
                True,
                ids[self.labels == self.good_label],
                ids[self.labels == self.bad_label],
            )

    def reject(self, X, direction):
        L = self.predict(X)
        target_label = self.good_label if direction == 0 else self.bad_label
        keep = X[L == target_label]
        return keep
