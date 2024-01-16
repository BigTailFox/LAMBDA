#!usr/bin/python
# coding=utf8
import math
import numpy as np

from .RegressionScissor import RegressionScissor
from .PlabelScissor import PlabelScissor
from .InfoScissor import InfoScissor
from .SampleBag import SampleBag
from .utils import mytimer


class Node:
    __n_nodes = 0

    def __init__(self, sample_bag: SampleBag, sample_idx=np.array([]), parent=None):
        self.parent = parent
        self.bag = sample_bag
        self.D_idx = np.array(sample_idx)
        self.depth = self.parent.depth + 1 if self.parent else 0
        self.kids = []
        # self.scissor = RegressionScissor()
        self.scissor = PlabelScissor()
        # self.scissor = InfoScissor()
        self.splitable = True
        self.fit_error = np.nan

        self.id = Node.__n_nodes
        Node.__n_nodes += 1

        self.n = np.nan
        self.mean = np.nan
        self.median = np.nan
        self.min = np.nan
        self.max = np.nan
        self.score = np.nan

        if len(self.D_idx) > 0:
            self.update_samples(self.D_idx)

    @staticmethod
    def reset_id():
        Node.__n_nodes = 0

    def try_split(self, weighted=False, min_lift=0):
        W = self.get_weight(normalize=True) if weighted else None
        samples = self.getD()
        success, good, bad = self.scissor.try_split(samples, self.D_idx, W, min_lift)
        if not success:
            self.splitable = False
            print("[WARNING] Node:try_split() failed on node {}!".format(self.id))
            return False, None, None
        else:
            self.fit_error = self.scissor.error
            return True, good, bad

    def get_score(self, cp, multi_beam=False, use_kde=False):
        if multi_beam and use_kde:
            self.score = self.get_uct_mbeam_kde(cp)
        elif not multi_beam and use_kde:
            self.score = self.get_uct_kde(cp)
        elif multi_beam and not use_kde:
            self.score = self.get_uct_mbeam(cp)
        elif not multi_beam and not use_kde:
            self.score = self.get_uct(cp)
        return self.score

    def get_loglikelihood(self):
        """
        返回节点内样本点上的对数似然概率
        """
        if not hasattr(self, "loglikelihood") or len(self.loglikelihood) != len(
            self.D_idx
        ):
            self.loglikelihood = self.bag.kde.score(self.getX())
        return self.loglikelihood

    def get_density(self, normalize=False):
        """
        返回节点内样本点上的概率密度
        """
        self.density = np.exp(self.get_loglikelihood())
        if normalize:
            self.density = self.density / np.sum(self.density)
        return self.density

    def get_weight(self, normalize=False):
        """
        返回节点内样本点的权重
        """
        self.weight = np.exp(-self.get_loglikelihood())
        if normalize:
            self.weight = self.weight / np.sum(self.weight)
        return self.weight

    def get_uct(self, cp):
        if self.parent == None or self.n == 0:
            return float("inf")
        exp_term = self.mean
        curiosity_term = 2 * math.sqrt(2 * math.log(self.parent.n, math.e) / self.n)
        uct = exp_term + cp * curiosity_term
        return uct

    def get_uct_mbeam(self, cp):
        if self.parent == None or self.n == 0:
            return float("inf")
        exp_term = self.mean
        curiosity_term = 2 * math.sqrt(2 * math.log(self.bag.num, math.e) / self.n)
        uct = exp_term + cp * curiosity_term
        return uct

    def get_uct_kde(self, cp):
        if self.parent == None or self.n == 0:
            return float("inf")
        w_node = self.get_weight(normalize=True)
        w = self.parent.get_weight(normalize=True)
        rou_node = np.sum(self.get_density() * w_node)
        rou = np.sum(self.parent.get_density() * w)
        exp_term = np.sum(self.getY() * w_node)
        curiosity_term = math.log(rou / rou_node, math.e)
        uct = exp_term + cp * curiosity_term
        return uct

    def get_uct_mbeam_kde(self, cp):
        if self.parent == None or self.n == 0:
            return float("inf")
        w_node = self.get_weight(normalize=True)
        w = self.bag.get_weight(normalize=True)
        rou_node = np.sum(self.get_density() * w_node)
        rou = np.sum(self.bag.get_density() * w)
        exp_term = np.sum(self.getY() * w_node)
        curiosity_term = math.log(rou / rou_node, math.e)
        uct = exp_term + cp * curiosity_term
        return uct

    def update_samples(self, idx):
        assert len(idx) <= self.bag.num
        self.D_idx = np.array(idx)
        self.n = len(self.D_idx)
        self.max = self.getY().max()
        self.min = self.getY().min()
        self.mean = self.getY().mean()
        # self.mean = np.sum(self.getY() * self.get_weight(True))
        self.median = np.median(self.getY())

    def is_leaf(self):
        if len(self.kids) == 0:
            return True
        return False

    def is_good_kid(self):
        if self.parent is None:
            return False
        if self.parent.kids[0] == self:
            return True
        else:
            return False

    def update_kids(self, good_kid_node, bad_kid_node):
        assert len(self.kids) == 0
        self.kids.append(good_kid_node)
        self.kids.append(bad_kid_node)

    def reject(self, X, direction):
        return self.scissor.reject(X, direction)

    def getX(self):
        rows = self.D_idx
        cols = self.bag.ds.c_features + self.bag.ds.d_features
        return self.bag.data.loc[rows, cols].to_numpy()

    def getD(self):
        rows = self.D_idx
        cols = self.bag.ds.c_features + self.bag.ds.d_features + [self.bag.ds.target]
        return self.bag.data.loc[rows, cols].to_numpy()

    def getY(self):
        rows = self.D_idx
        col = self.bag.ds.target
        return self.bag.data.loc[rows, col].to_numpy()

    def describe(self):
        print("--------------------[ Node{} ]--------------------")
        print("id:         ", self.id)
        print("parent:     ", self.parent.id if self.parent is not None else None)
        if self.parent is not None:
            print("is good kid?", self.is_good_kid())
        print("is leaf?    ", self.is_leaf())
        if not self.is_leaf():
            print("good kid:   ", self.kids[0].id if len(self.kids) >= 2 else None)
            print("bad kid:    ", self.kids[1].id if len(self.kids) >= 2 else None)
        print("n:          ", self.n)
        print("min:        ", self.min)
        print("mean:       ", self.mean)
        print("median:     ", self.median)
        print("max:        ", self.max)
        print("uct:        ", self.score)
        print("--------------------------------------------------")

    def __str__(self) -> str:
        return "<mcts.Node> Node{}".format(self.id)
