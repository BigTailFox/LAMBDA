#!usr/bin/python
# coding=utf8
import os
from .Replayer import Replayer
from mcts import DesignSpace, SampleBag, FunctionBase


class TwoParaScenario(Replayer):
    def __init__(self, name, verbose=False, lazy=True, method="linear"):
        design_space = DesignSpace(
            c_features={"distance": [10, 110], "velocity": [10, 30]},
            results={"minttc": [0, 12]},
            target="minttc",
        )
        sample_bag = SampleBag(design_space)
        data_path = f"{os.getenv('AD_EXPERIMENT_DATA')}/lattice_two_para.csv"
        super().__init__(
            name, design_space, sample_bag, verbose, lazy, data_path, method
        )
        self.load_data()
