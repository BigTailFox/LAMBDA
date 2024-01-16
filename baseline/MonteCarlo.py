#!usr/bin/python
# coding=utf8
from .Baseline import *
from mcts.Sampler import from_uniform


class MonteCarlo(Baseline):
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
        self.type = "MonteCarlo"

    def search(self, func: FunctionBase):
        X = from_uniform(self.total_evals, self.ds)
        while self.bag.num < self.total_evals:
            start = self.iter * self.iter_evals
            end = (self.iter + 1) * self.iter_evals
            start = max(0, start)
            end = min(self.total_evals, end)
            func.exe_batch(X[start:end, :])
            self.iter += 1
            self.metric.evaluate(self.iter)
            self.quick_plot(func.name)
            self.quick_save(func.name)
        self.quick_plot(func.name, force=True)
        self.quick_save(func.name, force=True)
