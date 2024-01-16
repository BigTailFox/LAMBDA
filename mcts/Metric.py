#!usr/bin/python
# coding=utf8
import os
import numpy as np
import pandas as pd

from .SampleBag import SampleBag
from .Config import ConfigMetric
from .FunctionBase import FunctionBase
from .utils import gen_pixels, draw_interpolate


def diff(frameA: np.ndarray, frameB: np.ndarray):
    """
    计算两个张量之间平方误差
    """
    assert frameA.shape == frameB.shape
    error = np.sum(np.power(frameA - frameB, 2))
    return error


def confusion_matrix_analysis(A, G, IR_filter, iteration, num):
    """
    混淆矩阵分析，对连续空间
    """
    assert A.shape == G.shape
    maskIR = IR_filter(G, True)
    maskNotIR = np.bitwise_not(maskIR)
    maskPredIR = IR_filter(A, True)
    maskPredNotIR = np.bitwise_not(maskPredIR)
    maskTP = np.bitwise_and(maskIR, maskPredIR)
    maskTN = np.bitwise_and(maskNotIR, maskPredNotIR)
    maskFP = np.bitwise_and(maskNotIR, maskPredIR)
    maskFN = np.bitwise_and(maskIR, maskPredNotIR)

    V = A.size
    P = np.count_nonzero(maskIR)
    N = V - P
    TP = np.count_nonzero(maskTP)
    TN = np.count_nonzero(maskTN)
    FP = np.count_nonzero(maskFP)
    FN = np.count_nonzero(maskFN)
    TPR = TP / P
    TNR = TN / N
    PREC = TP / (TP + FP) if TP + FP != 0 else np.nan

    ACC = (TP + TN) / V
    if PREC + TPR == 0:
        F1, F2, F05 = np.nan, np.nan, np.nan
    else:
        F1 = 2.0 * PREC * TPR / (PREC + TPR)
        F2 = 5.0 * PREC * TPR / (4.0 * PREC + TPR)
        F05 = 1.25 * PREC * TPR / (0.25 * PREC + TPR)

    error_of_IR = diff(A[maskIR], G[maskIR])
    error_r_of_IR = error_of_IR / np.sum(np.power(G[maskIR], 2))
    error = diff(A, G)
    error_r = error / np.sum(np.power(G, 2))

    s = pd.Series(
        {
            "num": num,
            "error": error_r,
            "error IR": error_r_of_IR,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "TPR": TPR,
            "TNR": TNR,
            "accuracy": ACC,
            "precision": PREC,
            "recall": TPR,
            "F1 score": F1,
            "F2 score": F2,
            "F0.5 score": F05,
        },
        name=iteration,
    )
    return s


class Metric(object):
    def __init__(self, bag: SampleBag, config: ConfigMetric = ConfigMetric()) -> None:
        super().__init__()
        self.bag = bag
        self.ds = self.bag.ds
        self.G = None
        self.config = config
        if type(self.config.dpi) is int:
            self.config.dpi = (self.config.dpi,) * (self.ds.cdim + self.ds.ddim)
        self.profile = pd.DataFrame(
            columns=[
                "num",
                "error",
                "error IR",
                "TP",
                "TN",
                "FP",
                "FN",
                "TPR",
                "TNR",
                "accuracy",
                "precision",
                "recall",
                "F1 score",
                "F2 score",
                "F0.5 score",
            ]
        )

    def get_groundtruth_from_file(self, filepath, inverse=False):
        _, Grids = gen_pixels(self.ds, self.config.dpi, returnGG=True)
        bag = SampleBag(self.ds)
        bag.load_csv2(filepath)
        sort_cols = self.ds.features[self.ds.cdim + self.ds.ddim :: -1]
        bag.data.sort_values(by=sort_cols, inplace=True)
        Z = -bag.getY() if inverse else bag.getY()
        assert Z.size == Grids[0].size
        self.G = Z.reshape(Grids[0].shape)
        return self.G

    def get_groundtruth_from_func(self, func: FunctionBase):
        pixels, Grids = gen_pixels(self.ds, self.config.dpi, returnGG=True)
        tempbag = func.bag
        func.bag = SampleBag(self.ds)
        func.lazy = True
        Z = func.exe_batch(pixels).reshape(Grids[0].shape)
        func.bag = tempbag
        self.G = -Z if self.config.inverse else Z
        return self.G

    def save_csv(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.profile.to_csv(folder + "profile.csv", index=True)
        print(
            "[INFO] saved {} entries to {}profile.csv".format(
                self.profile.shape[0], folder
            )
        )

    def evaluate(self, iteration):
        if self.bag.num < 10:
            return
        assert self.G is not None
        pixels, Grids = gen_pixels(self.ds, self.config.dpi, returnGG=True)
        A = draw_interpolate(
            self.bag.getX(), self.bag.getY(), pixels, self.config.interpolate_method
        )
        A = A.reshape(Grids[0].shape)
        A = -A if self.config.inverse else A
        s = confusion_matrix_analysis(A, self.G, self.filter, iteration, self.bag.num)
        self.profile = self.profile.append(s, ignore_index=False)
        print(
            "√ num: {}, precision: {:.3f}, recall: {:.3f}, F2 score: {:.3f}".format(
                self.bag.num, s["precision"], s["recall"], s["F2 score"]
            )
        )

    def filter(self, A, in_IR=True):
        rule = self.config.IR_rule[0]
        metric = self.config.IR_rule[1]
        if in_IR:
            if rule == "<":
                return A < metric
            elif rule == "<=":
                return A <= metric
            elif rule == "==":
                return A == metric
            elif rule == ">":
                return A > metric
            elif rule == ">=":
                return A >= metric
            elif rule == "in":
                return np.bitwise_and(A > metric[0], A < metric[1])
        else:
            return np.bitwise_not(self.filter(A, in_IR=True))
