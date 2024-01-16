#!usr/bin/python
# coding=utf8
import numpy as np
from .FunctionBase import FunctionBase
from .DesignSpace import DesignSpace
from .SampleBag import SampleBag


class Levy(FunctionBase):
    def __init__(
        self,
        name,
        design_space: DesignSpace,
        sample_bag: SampleBag,
        verbose=False,
        lazy=True,
    ):
        super().__init__(name, design_space, sample_bag, verbose=verbose, lazy=lazy)

    def calculate(self, x):
        w = 1 + (x - 1) / 4
        term1 = (np.sin(np.pi * w[0])) ** 2
        term2 = 0
        term3 = (w[-1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[-1])) ** 2)
        for idx in range(1, len(w)):
            wi = w[idx]
            new = (wi - 1) ** 2 * (1 + 10 * (np.sin(np.pi * wi + 1)) ** 2)
            term2 += new
        result = term1 + term2 + term3
        return result


class Rosenrock(FunctionBase):
    def __init__(
        self,
        name,
        design_space: DesignSpace,
        sample_bag: SampleBag,
        verbose=False,
        lazy=True,
    ):
        super().__init__(name, design_space, sample_bag, verbose=verbose, lazy=lazy)

    def calculate(self, x):
        result = sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
        return result


class Ackley(FunctionBase):
    def __init__(
        self,
        name,
        design_space: DesignSpace,
        sample_bag: SampleBag,
        verbose=False,
        lazy=True,
    ):
        super().__init__(name, design_space, sample_bag, verbose=verbose, lazy=lazy)

    def calculate(self, x):
        w = 1 + (x - 1) / 4
        result = (
            -20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size))
            - np.exp(np.cos(2 * np.pi * x).sum() / x.size)
            + 20
            + np.e
        )
        return result


class Rastrigin(FunctionBase):
    def __init__(
        self,
        name,
        design_space: DesignSpace,
        sample_bag: SampleBag,
        verbose=False,
        lazy=True,
    ):
        super().__init__(name, design_space, sample_bag, verbose=verbose, lazy=lazy)

    def calculate(self, x):
        tmp = 0
        for idx in range(0, len(x)):
            curt = x[idx]
            tmp = tmp + (curt**2 - 10 * np.cos(2 * np.pi * curt))
        result = 10 * len(x) + tmp
        return result


class Schwefel(FunctionBase):
    def __init__(
        self,
        name,
        design_space: DesignSpace,
        sample_bag: SampleBag,
        verbose=False,
        lazy=True,
    ):
        super().__init__(name, design_space, sample_bag, verbose=verbose, lazy=lazy)

    def calculate(self, x):
        result = 0
        for idx in range(0, len(x)):
            curt = x[idx]
            result = result + curt * np.sin(np.sqrt(np.abs(curt)))
        result = 418.9829 * len(x) - result
        return result


class Hart6(FunctionBase):
    def __init__(
        self,
        name,
        design_space: DesignSpace,
        sample_bag: SampleBag,
        verbose=False,
        lazy=True,
    ):
        super().__init__(name, design_space, sample_bag, verbose=verbose, lazy=lazy)

    def calculate(self, x):
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        P = (
            np.array(
                [
                    [1312, 1696, 5569, 124, 8283, 5886],
                    [2329, 4135, 8307, 3736, 1004, 9991],
                    [2348, 1451, 3522, 2883, 3047, 6650],
                    [4047, 8828, 8732, 5743, 1091, 381],
                ]
            )
            * 0.0001
        )
        outer = 0
        for i in range(0, 4):
            inner = 0
            for j in range(0, 6):
                xj = x[j]
                Aij = A[i, j]
                Pij = P[i, j]
                inner = inner + Aij * ((xj - Pij) ** 2)
            new = alpha[i] * np.exp(-inner)
            outer = outer + new
        y = -(2.58 + outer) / 1.94
        return y


class Booth(FunctionBase):
    def __init__(
        self,
        name,
        design_space: DesignSpace,
        sample_bag: SampleBag,
        verbose=False,
        lazy=True,
    ):
        super().__init__(name, design_space, sample_bag, verbose=verbose, lazy=lazy)

    def calculate(self, x):
        result = (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2
        return result


class Square(FunctionBase):
    def __init__(
        self,
        name,
        design_space: DesignSpace,
        sample_bag: SampleBag,
        verbose=False,
        lazy=True,
    ):
        super().__init__(name, design_space, sample_bag, verbose=verbose, lazy=lazy)

    def calculate(self, x):
        return np.inner(x, x)


class HoelderTable(FunctionBase):
    def __init__(
        self,
        name,
        design_space: DesignSpace,
        sample_bag: SampleBag,
        verbose=False,
        lazy=True,
    ):
        super().__init__(name, design_space, sample_bag, verbose=verbose, lazy=lazy)
        assert self.ds.cdim == self.ds.dim == 2

    def calculate(self, x):
        x1 = x[0]
        x2 = x[1]
        z = np.abs(
            np.sin(x1)
            * np.cos(x2)
            * np.exp(np.abs(1 - np.sqrt(x1**2 + x2**2) / np.pi))
        )
        return z


class TwoModal(FunctionBase):
    def __init__(
        self,
        name,
        design_space: DesignSpace,
        sample_bag: SampleBag,
        verbose=False,
        lazy=True,
    ):
        super().__init__(name, design_space, sample_bag, verbose=verbose, lazy=lazy)

    def calculate(self, x):
        modal1 = np.exp(
            -0.5 * np.dot(x + np.array([-4.0, 0.0]), x + np.array([-4.0, 0.0]))
        )
        modal2 = np.exp(
            -0.5 * np.dot(x + np.array([4.0, 0.0]), x + np.array([4.0, 0.0]))
        )
        z = modal1 + modal2
        return z


class FourModal(FunctionBase):
    def __init__(
        self,
        name,
        design_space: DesignSpace,
        sample_bag: SampleBag,
        verbose=False,
        lazy=True,
    ):
        super().__init__(name, design_space, sample_bag, verbose=verbose, lazy=lazy)

    def calculate(self, x):
        modal1 = np.exp(
            -0.5 * np.dot(x + np.array([-4.0, 0.0]), x + np.array([-4.0, 0.0]))
        )
        modal2 = np.exp(
            -0.5 * np.dot(x + np.array([4.0, 0.0]), x + np.array([4.0, 0.0]))
        )
        modal3 = np.exp(
            -0.5 * np.dot(x + np.array([0.0, -4.0]), x + np.array([0.0, -4.0]))
        )
        modal4 = np.exp(
            -0.5 * np.dot(x + np.array([0.0, 4.0]), x + np.array([0.0, 4.0]))
        )
        z = modal1 + modal2 + modal3 + modal4
        return z
