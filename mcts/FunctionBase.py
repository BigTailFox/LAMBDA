#!usr/bin/python
# coding=utf8
import numpy as np

from .SampleBag import SampleBag
from .DesignSpace import DesignSpace


class FunctionBase(object):
    def __init__(
        self,
        name,
        design_space: DesignSpace,
        sample_bag: SampleBag,
        verbose=False,
        lazy=False,
    ):
        assert design_space is not None or sample_bag is not None
        self.name = name
        self.verbose = verbose
        self.ds = sample_bag.design_space if design_space is None else design_space
        self.bag = SampleBag(self.ds) if sample_bag is None else sample_bag
        # to intergrate with some algorithms solving minimization problems.
        self.minimize_mode = False
        # when evaluate a batch of points, only track the results on the end, if lazy.
        # this would improve the performance of sample bag by reducing the pandas DataFrame constructions.
        self.lazy = lazy

    def describe(self):
        print("function:    ", self.name)
        print("verbose:     ", self.verbose)
        self.ds.describe()

    def calculate(self, x):
        """
        here the implementation of `calculate` must be a `maximize` form
        """
        print("[WARNING] default implementation of FunctionBase, return 0.")
        return 0

    def validate_input(self, x):
        assert len(x) == self.ds.dim
        assert np.all(x[: self.ds.cdim + self.ds.ddim] <= self.ds.ub) and np.all(
            x[: self.ds.cdim + self.ds.ddim] >= self.ds.lb
        )
        for i in range(self.ds.ndim):
            assert (
                x[self.ds.cdim + self.ds.ddim + i]
                in self.ds.n_map[self.ds.n_features[i]]
            )

    def track(self, X, Y):
        self.bag.append(X, Y)

    def __call__(self, x):
        self.validate_input(x)
        y = self.calculate(x)
        self.track(x, y)
        return -y if self.minimize_mode else y

    def exe_batch(self, X):
        Y = []
        if self.lazy:
            for x in X:
                self.validate_input(x)
                Y.append(self.calculate(x))
            Y = np.array(Y)
            self.track(X, Y)
            return -Y if self.minimize_mode else Y
        else:
            for x in X:
                Y.append(self.__call__(x))
            Y = np.array(Y)
            return Y
