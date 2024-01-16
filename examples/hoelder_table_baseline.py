#!usr/bin/python
# coding=utf8
import sys

sys.path.append("..")
from mcts.test_functions import *
from baseline import *
from experiments import TwoParaScenario, FDScenario, FICase1


def test_hoelder_table(test_num=0, algorithm="mc"):
    ds = DesignSpace(
        {"X": [-10, 10], "Y": [-10, 10]}, results={"R": [0, 20]}, target="R"
    )
    bag = SampleBag(ds)
    func = HoelderTable(
        "{}_HoelderTable_{}".format(algorithm.upper(), test_num), ds, bag, lazy=True
    )
    if algorithm in ["mc", "rs"]:
        opt = MonteCarlo(ds, bag, use_kde=True, iter_evals=50, total_evals=5000)
    elif algorithm in ["sobol"]:
        opt = SobolSequence(ds, bag, use_kde=True, iter_evals=50, total_evals=5000)
    elif algorithm in ["GA", "ga"]:
        opt = GAopt(ds, bag, use_kde=True, iter_evals=50, total_evals=5000)
    elif algorithm in ["DE", "de"]:
        opt = DEopt(ds, bag, use_kde=True, iter_evals=50, total_evals=5000)
    elif algorithm in ["BO", "bo"]:
        opt = BO(ds, bag, use_kde=True, iter_evals=50, total_evals=5000, init_evals=256)
    elif algorithm in ["TurBO", "TURBO", "turbo"]:
        opt = TurBO(
            ds,
            bag,
            use_kde=True,
            iter_evals=200,
            total_evals=5000,
            init_evals=64,
            n_trust_region=4,
        )
    opt.metric.get_groundtruth_from_func(func)
    opt.plotter.config.plot_curve_step = 1000
    opt.plotter.config.plot_density_step = 1000
    opt.plotter.config.plot_dynamics_step = 1000
    opt.plotter.config.plot_interpolate_step = 1000
    opt.metric.config.IR_rule = (">", 18)
    opt.metric.config.interpolate_method = "linear"
    opt.dump_step = 1000
    if algorithm.lower() in ["ga", "de", "bo", "turbo"]:
        func.minimize_mode = True
    opt.search(func)


for i in range(1, 11):
    test_hoelder_table(i, "mc")
for i in range(1, 11):
    test_hoelder_table(i, "sobol")
for i in range(1, 11):
    test_hoelder_table(i, "ga")
for i in range(1, 11):
    test_hoelder_table(i, "de")
for i in range(1, 11):
    test_hoelder_table(i, "bo")
for i in range(1, 11):
    test_hoelder_table(i, "turbo")
