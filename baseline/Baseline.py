#!usr/bin/python
# coding=utf8
import os
import pickle
from abc import ABC, abstractmethod
from mcts import DesignSpace, SampleBag, FunctionBase, Plotter, Metric


class Baseline(ABC):
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
        super().__init__()
        assert ds is not None or bag is not None
        self.ds = bag.ds if ds is None else ds
        self.bag = SampleBag(ds) if bag is None else bag
        self.metric = Metric(self.bag) if metric is None else metric
        self.plotter = Plotter(self.bag) if plotter is None else plotter
        self.dump_step = dump_step
        self.inverse = inverse
        self.metric.config.inverse = self.inverse
        self.bag.use_kde = use_kde
        self.iter_evals = iter_evals
        self.total_evals = total_evals

        self.iter = 0
        self.type = "BaselineBase"

    @abstractmethod
    def search(self, num, func: FunctionBase):
        pass

    def dump(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(folder + self.type, "wb") as f:
            pickle.dump(self, f)
        print("[INFO] dump {} to {}".format(folder, self.type))

    @staticmethod
    def load(folder, file):
        with open(folder + type, "rb") as f:
            agent = pickle.load(f)
        print("[INFO] load {} from {}".format(file, folder))
        return agent

    def quick_save(self, name, force=False):
        if force or (self.dump_step > 0 and self.iter % self.dump_step == 0):
            folder_path = "{}/{}/iter{}_point{}/".format(
                name, self.type, self.iter, self.bag.num
            )
            print("")
            # self.dump(folder_path)
            # self.bag.dump(folder_path)
            self.bag.save_csv(folder_path)
            self.metric.save_csv(folder_path)
            print("")

    def quick_plot(self, funcname, force=False):
        curve_step = self.plotter.config.plot_curve_step
        if force or (curve_step > 0 and self.iter % curve_step == 0):
            self.plotter.plot_curve(
                self.metric.profile,
                funcname,
                self.iter,
                self.metric.config.IR_rule,
                self.inverse,
            )

        intp_step = self.plotter.config.plot_interpolate_step
        intp_mode = self.plotter.config.plot_interpolate_mode
        if force or (intp_step > 0 and self.iter % intp_step == 0):
            self.plotter.plot_interpolate(
                funcname, intp_mode, self.iter, inverse=self.inverse
            )

        dyna_step = self.plotter.config.plot_dynamics_step
        if force or (dyna_step > 0 and self.iter % dyna_step == 0):
            self.plotter.plot_dynamics(funcname, self.iter)

        density_step = self.plotter.config.plot_density_step
        if self.bag.use_kde and (
            force or (density_step > 0 and self.iter % density_step == 0)
        ):
            self.plotter.plot_density(funcname, self.iter)
