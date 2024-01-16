#!usr/bin/python
# coding=utf8
import os
from sklearn.utils._testing import ignore_warnings
from matplotlib import colors, pyplot as plt
import numpy as np

from .SampleBag import SampleBag
from .utils import gen_pixels, draw_interpolate
from .Config import ConfigPlot


class Plotter:
    def __init__(self, bag: SampleBag, config: ConfigPlot = ConfigPlot()) -> None:
        self.bag = bag
        self.ds = bag.ds
        self.config = config
        if type(self.config.dpi) is int:
            self.config.dpi = (self.config.dpi,) * (self.ds.cdim + self.ds.ddim)
        self.config.norm.vmin = self.ds.r_lb[self.ds.results.index(self.ds.target)]
        self.config.norm.vmax = self.ds.r_ub[self.ds.results.index(self.ds.target)]

    def make_title(self, name, iters):
        if iters is not None:
            title = "{}, iter {}, point {}".format(name, iters, self.bag.num)
        else:
            title = "{}, point {}".format(name, self.bag.num)
        return title

    def make_filename(self, funcname, name, iters):
        if not os.path.exists(funcname):
            os.makedirs(funcname)
        if iters is not None:
            savefile = "{}{}_iter{}_point{}.jpeg".format(
                funcname, name, iters, self.bag.num
            )
        else:
            savefile = "{}{}_point{}.jpeg".format(funcname, name, self.bag.num)
        return savefile

    # @ignore_warnings
    def plot_interpolate(
        self,
        funcname,
        method="nearest",
        iters=None,
        XX=None,
        YY=None,
        Z=None,
        inverse=False,
    ):
        if self.ds.cdim + self.ds.ddim != 2:
            return
        print("--> plotting interpolate......")
        fig = plt.figure()
        if XX is None or YY is None or Z is None:
            pixels, GG = gen_pixels(self.ds, self.config.dpi, True)
            XX = GG[0]
            YY = GG[1]
            Z = draw_interpolate(self.bag.getX(), self.bag.getY(), pixels, method)
            Z = -Z.reshape(GG[0].shape) if inverse else Z.reshape(GG[0].shape)
        plt.contourf(
            XX,
            YY,
            Z,
            levels=self.config.levels,
            alpha=self.config.contourf_alpha,
            norm=self.config.norm,
            cmap=self.config.cmap,
        )
        plt.colorbar()
        plt.xticks()
        plt.yticks()
        plt.xlim(self.ds.lb[0], self.ds.ub[0])
        plt.ylim(self.ds.lb[1], self.ds.ub[1])
        plt.xlabel(self.ds.features[0])
        plt.ylabel(self.ds.features[1])
        plt.title(self.make_title("Interpolate", iters))
        fig.savefig(
            self.make_filename(funcname + "/interpolate/", "interpolate", iters),
            dpi=600,
        )
        plt.close(fig)

    # @ignore_warnings
    def plot_dynamics(self, funcname, iters=None, XX=None, YY=None, IR=None):
        if self.ds.cdim + self.ds.ddim != 2:
            return
        print("--> plotting sampling dynamics......")
        fig = plt.figure()
        X = self.bag.data.iloc[:, 0]
        Y = self.bag.data.iloc[:, 1]
        Z = self.bag.get_index()
        norm = colors.Normalize(0, self.bag.num)
        cmap = plt.cm.plasma
        if IR is not None and XX is not None and YY is not None:
            plt.contour(XX, YY, IR, levels=2, colors="k", linewidths=0.8)
        plt.scatter(X, Y, c=Z, s=20.0, alpha=0.3, norm=norm, cmap=cmap)
        plt.colorbar()
        plt.xticks()
        plt.yticks()
        plt.xlim(self.ds.lb[0], self.ds.ub[0])
        plt.ylim(self.ds.lb[1], self.ds.ub[1])
        plt.xlabel(self.ds.features[0])
        plt.ylabel(self.ds.features[1])
        plt.title(self.make_title("Sampling Dynamics", iters))
        fig.savefig(
            self.make_filename(funcname + "/dynamics/", "dynamics", iters), dpi=600
        )
        plt.close(fig)

    # @ignore_warnings
    def plot_density(self, funcname, iters=None):
        if self.ds.cdim + self.ds.ddim != 2:
            return
        print("--> plotting kde......")
        fig = plt.figure()
        X = self.bag.getX()
        Z = self.bag.get_density(normalize=True)
        Z = np.log(Z)
        # Z = self.bag.get_weight()
        cmap = plt.cm.plasma
        pixels, Grids = gen_pixels(self.ds, self.config.dpi, returnGG=True)
        pixels = draw_interpolate(X, Z, pixels, self.config.plot_interpolate_mode)
        plt.contourf(
            Grids[0],
            Grids[1],
            pixels.reshape(Grids[0].shape),
            levels=self.config.levels,
            cmap=cmap,
        )
        plt.colorbar()
        plt.xticks()
        plt.yticks()
        plt.xlim(self.ds.lb[0], self.ds.ub[0])
        plt.ylim(self.ds.lb[1], self.ds.ub[1])
        plt.xlabel(self.ds.features[0])
        plt.ylabel(self.ds.features[1])
        plt.title(self.make_title("Density", iters))
        fig.savefig(
            self.make_filename(funcname + "/density/", "density", iters), dpi=600
        )
        plt.close(fig)

    # @ignore_warnings()
    def plot_curve(self, profile, funcname, iters=None, IR_rule=None, inverse=False):
        print("--> plotting curve......")
        pltcfg = self.config
        X = profile.loc[:, "num"]
        Z = -self.bag.getY() if inverse else self.bag.getY()
        fig = plt.figure(figsize=(12, 6))
        ax = plt.subplot(111, facecolor="gainsboro")
        ax.set_xlim(0 - self.bag.num * 0.02, self.bag.num * 1.02)
        vmin = pltcfg.norm.vmin
        vmax = pltcfg.norm.vmax
        ax.set_ylim(vmin - 0.02 * (vmax - vmin), vmax + 0.02 * (vmax - vmin))
        if IR_rule is not None:
            fillX = [-self.bag.num * 0.02, self.bag.num * 1.02]
            if IR_rule[0] == "in":
                fillY1 = [IR_rule[1][0]] * 2
                fillY2 = [IR_rule[1][1]] * 2
            else:
                fillY1 = [IR_rule[1]] * 2
                if IR_rule[0] in ["<", "<="]:
                    fillY2 = [vmin - 0.02 * (vmax - vmin)] * 2
                elif IR_rule[0] in [">", ">="]:
                    fillY2 = [vmax + 0.02 * (vmax - vmin)] * 2
            ax.fill_between(fillX, fillY1, fillY2, facecolor="whitesmoke")
        ax.plot(
            self.bag.best_sample_id + 1,
            Z[self.bag.best_sample_id],
            color="indianred",
            label="global opt",
            linestyle="",
            marker="*",
            ms=10,
            alpha=1,
        )
        ax.scatter(
            self.bag.data.index + 1,
            Z,
            c=Z,
            norm=pltcfg.norm,
            cmap=pltcfg.cmap,
            s=pltcfg.scatter_s,
            alpha=pltcfg.scatter_alpha,
        )
        ax1 = ax.twinx()
        for idx_name, cw in pltcfg.curve_index_color_width.items():
            ax1.plot(
                X,
                profile.loc[:, idx_name],
                color=cw[0],
                label=idx_name,
                linewidth=cw[1],
                #  marker='x', ms=3
            )
        ax1.set_ylim(-0.02, 1.02)
        fig.legend(loc=2, bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes)
        ax.set_xlabel("num of samples")
        ax.set_ylabel("target function value")
        ax1.set_ylabel(", ".join(pltcfg.curve_index_color_width.keys()))
        plt.title(self.make_title("Convergency Curve", iters))
        fig.savefig(self.make_filename(funcname + "/curve/", "curve", iters), dpi=600)
        plt.close(fig)

    # @ignore_warnings
    def plot_partition(
        self, XX, YY, values, funcname, iters=None, num_new=0, selected_mask=None
    ):
        if self.ds.cdim + self.ds.ddim != 2:
            return
        print("--> plotting partition......")
        data = self.bag.data
        pltcfg = self.config
        fig = plt.figure()
        plt.contourf(
            XX,
            YY,
            values,
            levels=pltcfg.levels,
            alpha=pltcfg.contourf_alpha,
            norm=pltcfg.norm,
            cmap=pltcfg.cmap,
        )
        plt.colorbar()

        if pltcfg.plot_scatter:
            plt.scatter(
                data.iloc[:, 0],
                data.iloc[:, 1],
                s=pltcfg.scatter_s,
                alpha=pltcfg.scatter_alpha,
                c=data.loc[:, self.bag.target],
                norm=pltcfg.norm,
                cmap=pltcfg.cmap,
            )

        if pltcfg.plot_scatter_selected and num_new > 0:
            new_samples = data.tail(num_new)
            plt.scatter(
                new_samples.iloc[:, 0],
                new_samples.iloc[:, 1],
                c=new_samples.loc[:, self.bag.target],
                norm=pltcfg.norm,
                cmap=pltcfg.cmap,
                marker="o",
                s=15,
                alpha=1,
                linewidths=0.5,
                edgecolors="whitesmoke",
            )

        plt.xticks()
        plt.yticks()
        plt.xlim(self.ds.lb[0], self.ds.ub[0])
        plt.ylim(self.ds.lb[1], self.ds.ub[1])
        plt.xlabel(self.ds.features[0])
        plt.ylabel(self.ds.features[1])
        plt.title(
            self.make_title("Partition, {}".format(pltcfg.plot_partition_mode), iters)
        )
        plt.savefig(
            self.make_filename(funcname + "/partition/", "partition", iters), dpi=600
        )
        plt.close(fig)

        fig = plt.figure()
        if pltcfg.plot_selected and selected_mask is not None:
            plt.contourf(
                XX,
                YY,
                selected_mask,
                levels=max(2, selected_mask.max() - selected_mask.min()),
                cmap=plt.cm.hot,
                alpha=1.0,
            )
            plt.colorbar()
            if pltcfg.plot_scatter_selected and num_new > 0:
                new_samples = data.tail(num_new)
                plt.scatter(
                    new_samples.iloc[:, 0],
                    new_samples.iloc[:, 1],
                    c=new_samples.loc[:, self.bag.target],
                    norm=pltcfg.norm,
                    cmap=pltcfg.cmap,
                    marker="o",
                    s=15,
                    alpha=1,
                    linewidths=0.5,
                    edgecolors="whitesmoke",
                )
            plt.xticks()
            plt.yticks()
            plt.xlim(self.ds.lb[0], self.ds.ub[0])
            plt.ylim(self.ds.lb[1], self.ds.ub[1])
            plt.xlabel(self.ds.features[0])
            plt.ylabel(self.ds.features[1])
            plt.title(self.make_title("Selection Freq", iters))
            plt.savefig(
                self.make_filename(funcname + "/selection/", "selection", iters),
                dpi=600,
            )
            plt.close(fig)

    # @ignore_warnings
    def plot_scatter(
        self, funcname, X=None, Y=None, Z=None, iters=None, XX=None, YY=None, IR=None
    ):
        if self.ds.cdim + self.ds.ddim != 2:
            return
        print("--> plotting scatter......")
        fig = plt.figure()
        X = self.bag.data.iloc[:, 0] if X is None else X
        Y = self.bag.data.iloc[:, 1] if Y is None else Y
        Z = self.bag.getY() if Z is None else Z
        if IR is not None and XX is not None and YY is not None:
            plt.contour(XX, YY, IR, levels=2, colors="k", linewidths=0.8)
        plt.scatter(
            X,
            Y,
            c=Z,
            s=self.config.scatter_s,
            alpha=1,
            norm=self.config.norm,
            cmap=self.config.cmap,
        )
        plt.colorbar()
        plt.xticks()
        plt.yticks()
        plt.xlim(self.ds.lb[0], self.ds.ub[0])
        plt.ylim(self.ds.lb[1], self.ds.ub[1])
        plt.xlabel(self.ds.features[0])
        plt.ylabel(self.ds.features[1])
        plt.title(self.make_title("Scatter", iters))
        fig.savefig(
            self.make_filename(funcname + "/scatter/", "scatter", iters), dpi=600
        )
        plt.close(fig)
