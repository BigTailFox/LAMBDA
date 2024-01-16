#!usr/bin/python
# coding=utf8
import os
import pickle

import pandas as pd
import numpy as np

from .utils import mytimer
from .DensityEstimator import DensityEstimator
from .DesignSpace import DesignSpace


def index_update_scheduler(num, dim):
    if 0 <= num < 10 * dim:
        return 1
    elif 10 * dim <= num < 1000 * dim:
        return int(num * 0.1)
    elif 1000 * dim <= num:
        return int(num * 0.5)
    else:
        return float("inf")


class SampleBag:
    def __init__(self, design_space: DesignSpace) -> None:
        self.ds = design_space
        self.column_labels = self.ds.features + self.ds.results
        self.target = self.ds.target
        self.data = pd.DataFrame(columns=self.column_labels)
        self.num = 0
        self.best_sample_id = None
        self.best_sample_trace = pd.DataFrame(columns=self.column_labels)
        self.best = np.nan
        self.use_kde = False
        self.kde = DensityEstimator(self.ds)
        self.new_for_kde = 0

    def get_index(self):
        return self.data.index.tolist()

    def getX(self, copy=True):
        return self.data.loc[:, self.ds.features].to_numpy()

    def getY(self, copy=True):
        return self.data.loc[:, self.ds.target].to_numpy()

    def getD(self, copy=True):
        cols = self.ds.features + [self.ds.target]
        return self.data.loc[:, cols].to_numpy()

    def get_loglikelihood(self):
        """
        返回所有样本点上的对数似然概率
        """
        if not hasattr(self, "loglikelihood") or len(self.loglikelihood) != self.num:
            self.loglikelihood = self.kde.score(self.getX())
        return self.loglikelihood

    def get_density(self, normalize=False):
        """
        返回所有样本点上的分布密度
        """
        self.density = np.exp(self.get_loglikelihood())
        if normalize:
            self.density = self.density / np.sum(self.density)
        return self.density

    def get_weight(self, normalize=False):
        """
        返回所有样本点上的权重
        """
        self.weight = np.exp(-self.get_loglikelihood())
        if normalize:
            self.weight = self.weight / np.sum(self.weight)
        return self.weight

    def update_kde(self):
        """
        使用现有样本点更新核密度估计器
        """
        cols = self.ds.c_features + self.ds.d_features
        schedule = self.kde.config.update_period
        if schedule == "auto":
            schedule = index_update_scheduler(self.num, self.ds.cdim + self.ds.ddim)
        if self.new_for_kde >= schedule:
            X = self.data.loc[:, cols].to_numpy()
            self.kde.train(X)
            self.kde.add(X)
            self.new_for_kde = 0
        else:
            self.kde.add(self.data.loc[-self.new_for_kde :, cols].to_numpy())

    def append(self, X, Y=None) -> None:
        if Y is None:
            X = np.array(X).reshape(-1, self.ds.dim + self.ds.rdim)
        else:
            X = np.array(X).reshape(-1, self.ds.dim)
            Y = np.array(Y).reshape(-1, self.ds.rdim)
            assert len(X) == len(Y)
            X = np.concatenate([X, Y], axis=1)
        X = pd.DataFrame(X, columns=self.column_labels)
        self.data = pd.concat([self.data, X], ignore_index=True)
        self.num = self.data.shape[0]
        self.new_for_kde += len(X)
        self.update_best(X)
        if self.use_kde:
            self.update_kde()

    def update_best(self, X):
        for i in range(len(X)):
            if (
                self.best_sample_id is None
                or X.loc[i, self.target]
                > self.data.loc[self.best_sample_id, self.target]
            ):
                self.best_sample_id = self.num - len(X) + i
                self.best = self.data.loc[self.best_sample_id, self.target]
                self.best_sample_trace = self.best_sample_trace.append(
                    self.data.iloc[self.best_sample_id, :], ignore_index=False
                )

    def clear(self):
        self.data = pd.DataFrame(columns=self.column_labels)
        self.num = 0
        self.best_sample_id = None
        self.best_sample_trace = pd.DataFrame(columns=self.column_labels)
        self.best = np.nan

    def update_samples(self, X, Y=None):
        self.clear()
        self.append(X, Y)
        # self.update_best(X)

    def save_csv(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.data.to_csv(folder + "sample.csv", index=True)
        self.best_sample_trace.to_csv(folder + "trace.csv", index=True)
        print(
            "[INFO] saved {} samples to {}sample.csv".format(self.data.shape[0], folder)
        )

    def load_csv(self, folder):
        self.data = pd.read_csv(folder + "sample.csv", index_col=0)
        self.num = self.data.shape[0]
        self.best_sample_id = self.data.loc[:, self.target].values.argmax()
        self.best_sample_trace = pd.read_csv(folder + "trace.csv", index_col=0)
        print(
            "[INFO] load {} samples from {}samples.csv".format(
                self.data.shape[0], folder
            )
        )

    def load_csv2(self, file):
        self.data = pd.read_csv(file, index_col=0)
        self.num = self.data.shape[0]
        self.best_sample_id = self.data.loc[:, self.target].values.argmax()
        self.best_sample_trace = self.data[
            self.best_sample_id : self.best_sample_id + 1
        ]
        print("[INFO] load {} samples from {}".format(self.data.shape[0], file))

    def dump(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(folder + "SampleBagDump", "wb") as f:
            pickle.dump(self, f)
        print("[INFO] dump Sample Bag to {}SampleBagDump".format(folder))

    @staticmethod
    def load(folder):
        with open(folder + "SampleBagDump", "rb") as f:
            sample_bag = pickle.load(f)
        print("[INFO] load sample bag from {}SampleBagDump".format(folder))
        return sample_bag

    @staticmethod
    def load2(file):
        with open(file, "rb") as f:
            sample_bag = pickle.load(f)
        print("[INFO] load sample bag from {}".format(file))
        return sample_bag

    def describe(self):
        print("--------------------[ Data Bag ]--------------------")
        if self.data.shape[0] > 0:
            print("\nabstract of dataframe:")
            print(self.data.describe())
            print("current best sample id: ", self.best_sample_id)
            print("current best sample:")
            print(self.best_sample_trace.iloc[-1, :])
        else:
            print("empty dataframe")
        print("----------------------------------------------------")
