#!usr/bin/python
# coding=utf8
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.quasirandom import SobolEngine
from .FunctionBase import FunctionBase


def reject_sample(X, path):
    assert len(path) > 0
    total = len(X)
    for node, direction in path:
        if X.shape[0] == 0:
            return np.array([]), 0.0
        X = node.reject(X, direction)
    assert len(X) <= total
    return X, len(X) / total


def from_sobol(num, ds, seed=None):
    lb = ds.lb
    ub = ds.ub
    so = SobolEngine(ds.cdim + ds.ddim, scramble=True, seed=seed)
    X = so.draw(num).to(dtype=torch.float64).cpu().detach().numpy()
    X = X * (ub - lb) + lb
    return X


def from_uniform(num, ds):
    lb = ds.lb
    ub = ds.ub
    X = np.random.uniform(lb, ub, size=(num, ds.cdim + ds.ddim))
    return X


def from_norm(num, ds, mu, sigma):
    lb = ds.lb
    ub = ds.ub
    X = np.random.normal(mu, sigma, size=(num, ds.cdim + ds.ddim))
    X = np.clip(X, lb, ub)
    return X


def make_sampler(ds, method="sobol", path=None):
    assert method in ["sobol", "random", "uniform", "mc", "rs"]
    if method in ["random", "uniform", "mc", "rs"]:
        return UniformSampler(ds, path)
    elif method in ["sobol"]:
        return SobolSampler(ds, path)


class SamplerBase(ABC):
    def __init__(self, ds, path=None) -> None:
        super().__init__()
        self.ds = ds
        self.path = path

    def draw(self, num) -> np.ndarray:
        if self.path is None or not self.path:
            X = self._draw(num)
            return X
        cnt = 0
        batch = 1000
        X = np.array([]).reshape(-1, self.ds.cdim + self.ds.ddim)
        while cnt < num:
            _X = self._draw(batch)
            _X, ratio = reject_sample(_X, self.path)
            if ratio == 0.0:
                batch *= 2
                continue
            X = np.concatenate([X, _X], axis=0)
            cnt = len(X)
        if len(X) > num:
            X = X[np.random.choice(len(X), num)]
        return X

    def propose_and_sample(self, num, func: FunctionBase):
        X = self.draw(num)
        Y = func.exe_batch(X)
        return X, Y

    @abstractmethod
    def _draw(self, num) -> np.ndarray:
        pass


class UniformSampler(SamplerBase):
    def __init__(self, ds, path=None) -> None:
        super().__init__(ds, path=path)

    def _draw(self, num) -> np.ndarray:
        return from_uniform(num, self.ds)


class SobolSampler(SamplerBase):
    def __init__(self, ds, path=None, seed=None) -> None:
        super().__init__(ds, path=path)
        self.seed = seed

    def _draw(self, num) -> np.ndarray:
        return from_sobol(num, self.ds, self.seed)
