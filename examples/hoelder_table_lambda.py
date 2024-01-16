#!usr/bin/python
# coding=utf8
from mcts.test_functions import HoelderTable
from mcts import *


def test_heolder_table(tag="0", cp=0.1):
    ds = DesignSpace(
        {"X": [-10, 10], "Y": [-10, 10]}, results={"R": [0, 20]}, target="R"
    )
    bag = SampleBag(ds)
    func = HoelderTable("Lambda_HoelderTable_{}".format(tag), ds, bag, lazy=True)
    opt = MCTS(ds, bag)

    opt.config.verbose = False
    opt.config.dump_step = 10
    opt.config.max_depth = 8
    opt.config.leaf_size = 10
    opt.config.cp = cp * 20
    opt.config.use_kde = True
    opt.config.multi_beam = True
    opt.config.beam = 2
    opt.config.batch_evals = 1
    opt.config.total_evals = 5000
    opt.config.init_evals = 256
    opt.config.selects = 50
    opt.metric.config.IR_rule = (">", 18)

    bag.kde.config.k = 20

    opt.plotter.config.plot_curve_step = 20
    opt.plotter.config.plot_density_step = 20
    opt.plotter.config.plot_dynamics_step = 20
    opt.plotter.config.plot_partition_step = 20
    opt.plotter.config.plot_interpolate_step = 20
    opt.plotter.config.plot_partition_mode = "mean"
    opt.plotter.config.plot_scatter = False
    opt.plotter.config.plot_scatter_selected = False

    opt.metric.config.interpolate_method = "linear"
    opt.metric.get_groundtruth_from_func(func)
    opt.search(func)


for i in range(1, 11):
    test_heolder_table(str(i) + "_0.1", 0.1)
