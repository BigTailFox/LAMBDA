#!usr/bin/python
# coding=utf8
import sys

sys.path.append("..")
from mcts.test_functions import *
from baseline import *
from experiments import TwoParaScenario, FDScenario, FICase1


def test_two_para_scenario(test_num=0, algorithm="mc"):
    func = TwoParaScenario(
        "{}_TwoParaScenario_{}".format(algorithm.upper(), test_num), method="linear"
    )
    ds = func.ds
    bag = func.bag
    if algorithm in ["mc", "rs"]:
        opt = MonteCarlo(
            ds, bag, use_kde=True, inverse=True, iter_evals=50, total_evals=5000
        )
    elif algorithm in ["sobol"]:
        opt = SobolSequence(
            ds, bag, use_kde=True, inverse=True, iter_evals=50, total_evals=5000
        )
    elif algorithm in ["GA", "ga"]:
        opt = GAopt(
            ds, bag, use_kde=True, inverse=True, iter_evals=50, total_evals=5000
        )
    elif algorithm in ["DE", "de"]:
        opt = DEopt(
            ds, bag, use_kde=True, inverse=True, iter_evals=50, total_evals=5000
        )
    elif algorithm in ["BO", "bo"]:
        opt = BO(
            ds,
            bag,
            use_kde=True,
            inverse=True,
            iter_evals=50,
            total_evals=5000,
            init_evals=32,
        )
    elif algorithm in ["TurBO", "TURBO", "turbo"]:
        opt = TurBO(
            ds,
            bag,
            use_kde=True,
            inverse=True,
            iter_evals=200,
            total_evals=5000,
            init_evals=8,
            n_trust_region=4,
        )
    opt.metric.get_groundtruth_from_func(func)
    opt.plotter.config.plot_curve_step = 10
    opt.plotter.config.plot_density_step = 10
    opt.plotter.config.plot_dynamics_step = 10
    opt.plotter.config.plot_interpolate_step = 10
    opt.metric.config.IR_rule = ("<", 0.5)
    opt.metric.config.interpolate_method = "linear"
    opt.dump_step = 10000
    if algorithm.lower() in ["ga", "de", "bo", "turbo"]:
        func.minimize_mode = True
    opt.search(func)


for i in range(1, 11):
    test_two_para_scenario(i, "mc")
for i in range(1, 11):
    test_two_para_scenario(i, "sobol")
for i in range(1, 11):
    test_two_para_scenario(i, "ga")
for i in range(1, 11):
    test_two_para_scenario(i, "de")
for i in range(1, 11):
    test_two_para_scenario(i, "turbo")
for i in range(1, 11):
    test_two_para_scenario(i, "bo")
