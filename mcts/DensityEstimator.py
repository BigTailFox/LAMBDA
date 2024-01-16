#!usr/bin/python
# coding=utf8

import faiss
import numpy as np
from sklearn.preprocessing import StandardScaler

from .DesignSpace import DesignSpace
from .Config import ConfigKDE
from .kernels import make_kernel


def make_index(key, dim):
    if key == "Flat L2":
        return faiss.IndexFlatL2(dim)


class DensityEstimator(object):
    def __init__(self, ds: DesignSpace, config: ConfigKDE = ConfigKDE()) -> None:
        super().__init__()
        self.ds = ds
        self.dim = self.ds.cdim + self.ds.ddim
        self.config = config
        self.kernel = make_kernel(self.config.kernel_type)
        self.scaler = StandardScaler().fit([self.ds.lb, self.ds.ub])
        self.indexer = make_index(self.config.index_type, self.dim)

    def to_unit(self, X):
        _X = self.scaler.transform(X).astype("float32")
        return _X.copy(order="C")

    def train(self, X):
        if self.indexer.is_trained:
            self.indexer = make_index(self.config.index_type, self.dim)
        self.indexer.train(self.to_unit(X))

    def add(self, X):
        assert self.indexer.is_trained
        self.indexer.add(self.to_unit(X))

    def search(self, X, k):
        assert self.indexer.is_trained and self.indexer.ntotal > 0
        D, I = self.indexer.search(self.to_unit(X), k)
        return D, I

    def score(self, X, k=None):
        """
        `X` : n samples for n rows, m features for m columns
        """
        assert self.indexer.ntotal > 0
        dim = X.shape[1]
        # 当索引中的向量不满足 k 时，少取一点
        k = min(self.indexer.ntotal, self.config.k if k is None else k)
        D, I = self.search(X, k)
        B = np.max(D, axis=1)  # adaptive bandwidth
        while np.count_nonzero(B) != len(B):
            k = min(self.indexer.ntotal, int(k * 1.5))
            print("[WARNING] duplicated points in index, enlarge k: {}".format(k))
            D, I = self.search(X, k)
            B = np.max(D, axis=1)
        D = D / B.reshape(-1, 1)
        P1 = np.mean(self.kernel(D, dim), axis=1)
        logP = np.log(P1) - dim * np.log(B)
        # assert np.all(logP < 0)
        return logP.astype("float64")
