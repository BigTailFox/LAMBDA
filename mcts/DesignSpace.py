#!usr/bin/python
# coding=utf8
import numpy as np


class DesignSpace(object):
    def __init__(
        self, c_features={}, d_features={}, n_features={}, results={}, target="default"
    ) -> None:
        """
        Args:
            `c_features`: { "X": [ lb, ub ] , ... }

            `d_features`: { "Y": [ lb, ub, dpi ] , ... }

            `n_features`: { "Z": [ 'A', 'B', 'C' ] , ... }

            `results`: { "result": [ lb, ub ] , ... }
        """
        super().__init__()
        self.c_features = list(c_features.keys())
        self.d_features = list(d_features.keys())
        self.n_features = list(n_features.keys())
        self.n_map = n_features
        self.features = self.c_features + self.d_features + self.n_features

        self.r_lb, self.r_ub, _ = self.parse_value(results.values())
        assert np.all(self.r_lb < self.r_ub)
        self.c_lb, self.c_ub, _ = self.parse_value(c_features.values())
        self.c_dpi = np.array([-1] * len(self.c_features), dtype="int")
        assert np.all(self.c_lb < self.c_ub)
        self.d_lb, self.d_ub, self.d_dpi = self.parse_value(d_features.values())
        assert np.all(self.d_lb < self.d_ub)
        assert np.all(self.d_dpi >= 2)
        self.lb = np.concatenate([self.c_lb, self.d_lb])
        self.ub = np.concatenate([self.c_ub, self.d_ub])
        self.dpi = np.concatenate([self.c_dpi, self.d_dpi])
        self.cdim = len(self.c_features)
        self.ddim = len(self.d_features)
        self.ndim = len(self.n_features)
        self.dim = self.cdim + self.ddim + self.ndim

        self.results = list(results.keys())
        self.target = self.results[-1] if target == "default" else target
        assert self.target in self.results
        self.rdim = len(self.results)
        assert self.rdim >= 1

    def describe(self):
        print(
            "dimensions:  total {}, c {}, d {}, n {}".format(
                self.dim, self.cdim, self.ddim, self.ndim
            )
        )
        print("c_features:  ", self.c_features)
        print("d_features:  ", self.d_features)
        print("n_features:  ", self.n_features)
        print("lower bound: ", self.lb)
        print("upper bound: ", self.ub)
        print("dpi:         ", self.dpi)
        print("n_map: ")
        print(self.n_map)
        print("results:     ", self.results)
        print("target:      ", self.target)
        print("results lb:  ", self.r_lb)
        print("results ub:  ", self.r_ub)

    def parse_value(self, values):
        lb = []
        ub = []
        dpi = []
        for v in values:
            if len(v) >= 1:
                lb.append(v[0])
            if len(v) >= 2:
                ub.append(v[1])
            if len(v) >= 3:
                dpi.append(v[2])
        lb = np.array(lb, dtype="float64")
        ub = np.array(ub, dtype="float64")
        dpi = np.array(dpi, dtype="int")
        return lb, ub, dpi
