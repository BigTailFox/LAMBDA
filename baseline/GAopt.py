#!usr/bin/python
# coding=utf8
import math
from .Baseline import *
from sko.GA import GA


class GAopt(Baseline):
    def __init__(
        self,
        ds: DesignSpace = None,
        bag: SampleBag = None,
        metric: Metric = None,
        plotter: Plotter = None,
        inverse=False,
        dump_step=100,
        use_kde=False,
        iter_evals=1,
        total_evals=400,
    ) -> None:
        super().__init__(
            ds=ds,
            bag=bag,
            metric=metric,
            plotter=plotter,
            inverse=inverse,
            dump_step=dump_step,
            use_kde=use_kde,
            iter_evals=iter_evals,
            total_evals=total_evals,
        )
        self.type = "GA"

    def search(self, func: FunctionBase):
        self.ga = GA(
            func,
            self.ds.cdim + self.ds.ddim,
            size_pop=self.iter_evals,
            max_iter=math.ceil(self.total_evals / self.iter_evals),
            lb=self.ds.lb,
            ub=self.ds.ub,
        )
        while self.bag.num < self.total_evals:
            self.ga.run(max_iter=1)
            self.iter += 1
            self.metric.evaluate(self.iter)
            self.quick_plot(func.name)
            self.quick_save(func.name)
        self.quick_plot(func.name, force=True)
        self.quick_save(func.name, force=True)
