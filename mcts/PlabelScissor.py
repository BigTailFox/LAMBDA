#!usr/bin/python
# coding=utf8
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class PlabelScissor(object):
    def __init__(self) -> None:
        super().__init__()
        self.scaler = StandardScaler()
        self.kmeans = KMeans(2)
        self.svc = SVC(C=1, kernel="linear", gamma="scale", class_weight="balanced")
        self.good_mean = np.nan
        self.bad_mean = np.nan
        self.good_label = 1
        self.bad_label = 0
        self.plabels = np.array([])
        self.labels = np.array([])
        self.error = np.nan
        self.mean = np.nan
        self.debug = False

    def get_mean(self, Y, W):
        return Y.mean() if W is None else np.sum(Y * W)

    def train(self, X, Y, W):
        W = None  # kmeans-svm 划分暂时不适用密度加权
        data = np.concatenate([X, Y.reshape(-1, 1)], axis=1)
        temp_scaler = StandardScaler()
        data = temp_scaler.fit_transform(data)
        self.plabels = self.kmeans.fit_predict(data, sample_weight=W)
        assert (
            len(np.unique(self.plabels)) == 2
        ), "PlabelScissor:train() train kmeans failed!"
        X = self.scaler.fit_transform(X)
        self.svc.fit(X, self.plabels, sample_weight=W)
        self.mean = self.get_mean(Y, W)

    def predict(self, X, returnY=False):
        X = self.scaler.transform(X)
        L = self.svc.predict(X)
        if returnY:
            return L, self.svc.decision_function(X)
        return L

    def try_split(self, samples, ids, weights=None, min_lift=0, returnY=False):
        X = samples[:, :-1]
        Y = samples[:, -1]
        W = weights
        self.train(X, Y, W)
        if returnY:
            self.labels, Yhat = self.predict(X, returnY=True)
        else:
            self.labels = self.predict(X)
        if len(np.unique(self.labels)) < 2:
            if self.debug:
                print(
                    "[DEBUG] PlabelScissor.try_split() failed, all points with one label"
                )
                print(
                    "        kmeans: label_0 {:4}, label_1 {:4}".format(
                        np.count_nonzero(self.plabels == 0),
                        np.count_nonzero(self.plabels == 1),
                    )
                )
                print(
                    "           svm: label_0 {:4}, label_1 {:4}".format(
                        np.count_nonzero(self.labels == 0),
                        np.count_nonzero(self.labels == 1),
                    )
                )
            if returnY:
                return False, None, None, Yhat
            else:
                return False, None, None
        self.error = self.get_error()
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
                print(
                    "[DEBUG] PlabelScissor:try_split() failed, not enough lift on mean"
                )
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

    def get_error(self):
        TP = np.bitwise_and(self.labels == self.plabels, self.plabels == 0)
        TN = np.bitwise_and(self.labels == self.plabels, self.plabels == 1)
        FP = np.bitwise_and(self.labels != self.plabels, self.plabels == 1)
        FN = np.bitwise_and(self.labels != self.plabels, self.plabels == 0)
        TP = np.count_nonzero(TP)
        TN = np.count_nonzero(TN)
        FP = np.count_nonzero(FP)
        FN = np.count_nonzero(FN)
        recall = TP / (FP + TP) if FP + TP != 0 else np.nan
        precision = TP / (TP + FN) if TP + FN != 0 else np.nan
        f1 = 2.0 * precision * recall / (precision + recall)
        return 1 - f1
