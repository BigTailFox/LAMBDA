#!usr/bin/python
# coding=utf8
from .Baseline import *
from .turbo.bo import BayesOpt


class BO(Baseline):
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
        init_evals=100,
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
        self.type = "BO"
        self.init_evals = init_evals

    def search(self, func: FunctionBase):
        self.bo = BayesOpt(
            func,
            self.ds.lb,
            self.ds.ub,
            self.init_evals,
            self.total_evals,
            self.iter_evals,
            verbose=False,
        )
        while self.bag.num < self.total_evals:
            self.bo.optimize(1)
            self.iter += 1
            self.metric.evaluate(self.iter)
            self.quick_plot(func.name)
            self.quick_save(func.name)
        self.quick_plot(func.name, force=True)
        self.quick_save(func.name, force=True)
