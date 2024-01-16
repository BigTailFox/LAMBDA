#!usr/bin/python
# coding=utf8
import sys

import matplotlib
sys.path.append("../../")
from mcts.utils import mytimer, gen_pixels
import seaborn as sns
from mcts.test_functions import HoelderTable
from mcts.Config import ConfigMetric, ConfigPlot
from mcts import DesignSpace, SampleBag, Metric, Plotter
from matplotlib import pyplot as plt
from matplotlib import colors as cm
import pandas as pd
import numpy as np


ds = DesignSpace({"X": [-10, 10], "Y": [-10, 10]},
                 results={"R": [0, 20]}, target="R")
config = ConfigMetric()
config.dpi = (100, 100)
config.interpolate_method = 'linear'
config.IR_rule = ('>', 18)
func = HoelderTable("_ht", ds, SampleBag(ds))
_metric = Metric(func.bag, config)
_metric.get_groundtruth_from_func(func)

plt.rcParams['font.sans-serif'] = 'Times New Roman'
fontsize1 = 18
fontsize2 = 14
fontsize3 = 18

def read_data(algorithm, i, points):
    bag = SampleBag(ds)
    path = "{}/{}/sample.csv".format(algorithm, i)
    bag.load_csv2(path)

    bagm = SampleBag(ds)
    metric = Metric(bagm, config)
    metric.G = _metric.G

    iters = 1
    while bagm.num < points:
        start = (iters - 1) * 10
        end = min(iters * 10, bag.num)
        bagm.append(bag.data.iloc[start:end, :].to_numpy())
        metric.evaluate(i)
        iters += 1
    return metric.profile.loc[:, ['num', 'F2 score']].rename(columns={'F2 score': i})


def read_batch(algorithm, num, points):
    frames = None
    for i in range(1, num+1):
        if frames is None:
            frames = read_data(algorithm, i, points)
        else:
            new = read_data(algorithm, i, points)
            frames = pd.merge(frames, new, on='num')
    frames.to_csv('{}.csv'.format(algorithm.lower()))
    return frames


def resample_all():
    read_batch('MC', 10, 5000)
    read_batch('SOBOL', 10, 5000)
    read_batch('DE', 10, 5000)
    read_batch('GA', 10, 5000)
    read_batch('TURBO', 10, 5000)
    read_batch('LA', 10, 5000)
    read_batch('Lambda', 10, 5000)
    read_batch('BO', 10, 5000)


def plot_sampling_dynamics():
    algs = ['mc', 'sobol', 'de', 'ga', 'bo', 'turbo', 'la', 'lambda']
    labels = ['Random', 'Sobol', 'DE', 'GA', 'BO',
              'TuRBO', 'LaMCTS', 'LAMBDA (ours)']
    fig = plt.figure(figsize=(16,8))
    for i in range(len(algs)):
        ax = plt.subplot(2, 4, i+1)
        alg = algs[i]
        label = labels[i]
        bag = SampleBag(ds)
        path = "{}/{}/sample.csv".format(alg, 1)
        bag.load_csv2(path)
        bag.data = bag.data.iloc[:1500, :]
        X = bag.data.iloc[:, 0]
        Y = bag.data.iloc[:, 1]
        Z = bag.get_index()
        norm = matplotlib.colors.Normalize(0, len(Z))
        cmap = plt.cm.plasma
        ax.scatter(X, Y, c=Z,
                    s=20.0, alpha=0.3,
                    norm=norm, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(bag.ds.lb[0], bag.ds.ub[0])
        ax.set_ylim(bag.ds.lb[1], bag.ds.ub[1])
        # ax.set_xlabel(bag.ds.features[0])
        # ax.set_ylabel(bag.ds.features[1])
        ax.set_title(label, fontsize=fontsize1)
    # TODO: add colorbar
    plt.savefig('sampling_dynamics_hoelder_table.pdf', dpi=600)
    plt.savefig('sampling_dynamics_hoelder_table.jpeg', dpi=600)
    plt.close(fig)


def plot_density():
    algs = ['mc', 'sobol', 'de', 'ga', 'bo', 'turbo', 'la', 'lambda']
    labels = ['Random', 'Sobol', 'DE', 'GA', 'BO',
              'TuRBO', 'LaMCTS', 'LAMBDA (ours)']
    fig = plt.figure(figsize=(16, 8))
    for i in range(len(algs)):
        ax = plt.subplot(2, 4, i+1)
        alg = algs[i]
        label = labels[i]
        bag = SampleBag(ds)
        path = "{}/{}/sample.csv".format(alg, 1)
        bag.load_csv2(path)
        bag.data = bag.data.iloc[:1500, :]
        bag.use_kde = True
        bag.update_samples(bag.data.to_numpy())
        pixels, Grids = gen_pixels(bag.ds, (100, 100), returnGG=True)
        pixels = bag.kde.score(pixels, 20)
        pixels = np.exp(pixels)
        pixels = pixels / pixels.sum()
        cmap = plt.cm.plasma
        ax.contourf(Grids[0], Grids[1],
                    pixels.reshape(Grids[0].shape),
                    cmap=cmap,
                    norm=cm.LogNorm(),
                    )
        # ax.colorbar()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(bag.ds.lb[0], bag.ds.ub[0])
        ax.set_ylim(bag.ds.lb[1], bag.ds.ub[1])
        # ax.set_xlabel(bag.ds.features[0])
        # ax.set_ylabel(bag.ds.features[1])
        ax.set_title(label, fontsize=fontsize1)
    # TODO: add colorbar
    plt.savefig('density_hoelder_table.pdf', dpi=600)
    plt.savefig('density_hoelder_table.jpeg', dpi=600)
    plt.close(fig)


def plot_groundtruth():
    pixels, Grids = gen_pixels(ds, (300, 300), returnGG=True)
    pixels = func.exe_batch(pixels)
    pixels = pixels.reshape(Grids[0].shape)
    mask = pixels > 18
    fig = plt.figure(figsize=(10, 8))
    plotter = Plotter(func.bag)
    plt.contourf(Grids[0], Grids[1], pixels,
                 cmap=plotter.config.cmap,
                 norm=plotter.config.norm,
                 levels=25)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=20)
    cb.ax.set_title("f(x)", fontsize=fontsize3)
    plt.contour(Grids[0], Grids[1], mask, levels=2, colors='indianred', linewidths=1.2)
    plt.xlim(ds.lb[0], ds.ub[0])
    plt.ylim(ds.lb[1], ds.ub[1])
    plt.xlabel(ds.features[0], fontsize=fontsize1)
    plt.ylabel(ds.features[1], fontsize=fontsize1)
    plt.xticks(fontsize=fontsize1)
    plt.yticks(fontsize=fontsize1)
    # plt.title("Groundtruth of Hoelder Table Function", fontsize=20)
    plt.savefig('groundtruth_hoelder_table.pdf', dpi=600)
    plt.savefig('groundtruth_hoelder_table.jpeg', dpi=600)
    plt.close()

def plot():
    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(1, 1, 1)
    algs = ['mc', 'sobol', 'ga', 'de', 'bo', 'turbo', 'la', 'lambda']
    labels = ['Random', 'Sobol', 'DE', 'GA', 'BO',
              'TuRBO', 'LaMCTS', 'LAMBDA (ours)']
    # palette = sns.color_palette("hls", 8)
    # palette = sns.color_palette("Paired",8)
    palette = sns.hls_palette(8, l=.5, s=.6)
    # colors = ['steelblue','indianred','olivedrab', 'sienna',  'orange',
    #            'teal', 'royalblue', 'darkviolet']
    interval = 5
    for i in range(len(algs)):
    # for i in [0, 7]:
        alg = algs[i]
        # color = colors[i]
        color = palette[i]
        label = labels[i]
        df = pd.read_csv('{}.csv'.format(alg), index_col=0)
        df.set_index(df.num, drop=True, inplace=True)
        df.drop(columns=['num'], inplace=True)
        # print(df)
        ax.fill_between(df.index[::interval],
                        df.min(axis=1)[::interval],
                        df.max(axis=1)[::interval],
                        alpha=0.1,
                        color=color,
                        )
        ax.plot(df.index[::interval],
                df.mean(axis=1)[::interval],
                color=color,
                label=label,
                linewidth=1.5)
    ax.set_xlim(0, 5000)
    ax.set_ylim(0, 1)
    ax.set_xlabel("samples", fontsize=fontsize1)
    ax.set_ylabel("F2-score", fontsize=fontsize1)
    # ax.set_ylabel("coverage")
    ax.legend(loc="lower right", fontsize=fontsize2, ncol=2)
    plt.xticks(fontsize=fontsize2)
    plt.yticks(fontsize=fontsize2)
    ax.grid()
    # ax.set_title("F2-score Benchmark on Hoelder Table Function")
    # ax.set_title("Benchmark on Hoelder Table Function")
    # plt.savefig('benchmark_heolder_table_lambda_mc.jpeg', dpi=600)
    plt.savefig('benchmark_hoelder_table.pdf', dpi=600)
    plt.savefig('benchmark_hoelder_table.jpeg', dpi=600)
    plt.close()


# resample_all()
plot()
# plot_sampling_dynamics()
# plot_density()
# plot_groundtruth()
