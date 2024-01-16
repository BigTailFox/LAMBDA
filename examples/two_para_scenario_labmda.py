#!usr/bin/python
# coding=utf8
from mcts import *
from experiments import TwoParaScenario


def test_two_para(tag="0", cp=0.15):
    func = TwoParaScenario("Lambda_TwoParaScenario_{}".format(tag), method="linear")
    ds = func.ds
    bag = func.bag

    opt = MCTS(ds, bag)
    opt.config.verbose = False
    opt.config.max_depth = 6
    opt.config.leaf_size = 4
    opt.config.cp = cp * 10
    opt.config.use_kde = True
    opt.config.multi_beam = True
    opt.config.beam = 2
    opt.config.batch_evals = 1
    opt.config.total_evals = 5000
    opt.config.init_evals = 32
    opt.config.selects = 50
    opt.config.inverse = True
    opt.metric.config.inverse = True
    opt.metric.config.IR_rule = ("<", 0.5)

    bag.kde.config.k = 20

    step = 4
    opt.config.dump_step = 1000
    opt.plotter.config.plot_curve_step = step
    opt.plotter.config.plot_density_step = step
    opt.plotter.config.plot_dynamics_step = step
    opt.plotter.config.plot_partition_step = step
    opt.plotter.config.plot_interpolate_step = step
    opt.plotter.config.plot_partition_mode = "mean"
    opt.plotter.config.plot_interpolate_mode = "nearest"
    opt.plotter.config.plot_scatter = False
    opt.plotter.config.plot_scatter_selected = False

    opt.metric.config.interpolate_method = "linear"
    opt.metric.get_groundtruth_from_func(func)
    opt.search(func)


for i in range(1, 11):
    test_two_para(i, 0.05)
