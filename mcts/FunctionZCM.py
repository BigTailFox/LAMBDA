from .FunctionBase import FunctionBase
from .DesignSpace import DesignSpace
from .SampleBag import SampleBag
from .task_center.Task import Task
import sim_signal
import numpy as np


class FunctionZCM(FunctionBase):
    def __init__(
        self,
        name,
        design_space: DesignSpace,
        sample_bag: SampleBag,
        verbose=False,
        lazy=False,
    ):
        super().__init__(name, design_space, sample_bag, verbose=verbose, lazy=lazy)

    def calculate(self, x):
        task = Task()
        scenario = sim_signal.Scenario()
        scenario.flag = 0
        fi = sim_signal.FI()
        fi.flag = 1
        fi.index = self.bag.num
        fi.values = x
        fi.size = len(fi.values)
        ret = task.execute(scenario, fi, self.bag.num)
        return np.array(ret.values).reshape(-1)
