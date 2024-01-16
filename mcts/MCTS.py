#!usr/bin/python
# coding=utf8
import os
import pickle
import json
import numpy as np

from .Node import Node
from .DesignSpace import DesignSpace
from .SampleBag import SampleBag
from .FunctionBase import FunctionBase
from .Sampler import make_sampler
from .Config import ConfigMCTS
from .Metric import Metric
from .Plotter import Plotter
from .utils import mytimer


class MCTS:
    def __init__(
        self,
        design_space: DesignSpace = None,
        sample_bag: SampleBag = None,
        config: ConfigMCTS = ConfigMCTS(),
        metric: Metric = None,
        plotter: Plotter = None,
    ):
        self.config = config
        assert not (design_space is None and sample_bag is None)
        if design_space is None:
            self.ds = sample_bag.ds
        else:
            self.ds = design_space
        if sample_bag is None:
            self.bag = SampleBag(design_space)
        else:
            self.bag = sample_bag
        if metric is None:
            self.metric = Metric(self.bag)
        else:
            self.metric = metric
        self.metric.config.inverse = self.config.inverse
        if plotter is None:
            self.plotter = Plotter(self.bag)
        else:
            self.plotter = plotter

        self.iter = 0
        self.selects = 0  # 搜索开始总的 #selects
        self.nodes = []
        self.root = Node(self.bag, self.bag.get_index())
        self.nodes.append(self.root)

        self.num_new = 0  # 从当前树划分完成起新采集的样本点数
        self.selected_nodes = []  # 当前轮次被选中的节点
        self.selected_mask = None  # 当前轮次被选中的区域

    def get_leaf_status(self):
        status = []
        for node in self.nodes:
            temp = (
                node.is_leaf()
                and node.splitable
                and len(node.D_idx) >= self.config.leaf_size
                and node.depth < self.config.max_depth
            )
            if temp:
                status.append(True)
            else:
                status.append(False)
        return np.array(status)

    def get_leafs(self):
        leafs = []
        for node in self.nodes:
            if node.is_leaf():
                leafs.append(node)
        return leafs

    def nodes_to_split(self):
        status = self.get_leaf_status()
        node_ids = np.argwhere(np.array(status) == True).reshape(-1)
        return node_ids

    def should_treeify_continue(self):
        status = self.get_leaf_status()
        if True in status:
            return True
        return False

    def init_train(self, func: FunctionBase):
        sampler_type = self.config.global_sampler_type
        init_evals = self.config.init_evals
        print("\n★ init with {} {} points".format(sampler_type, init_evals))
        sampler = make_sampler(self.ds, sampler_type)
        sampler.propose_and_sample(init_evals, func)

    def tree_destroy(self):
        self.nodes.clear()
        self.num_new = 0
        self.selected_nodes = []
        self.selected_mask = None
        Node.reset_id()
        self.root = Node(self.bag, self.bag.get_index())
        self.nodes.append(self.root)

    # @mytimer
    def treeify(self):
        print(
            "\n★ LA-MCTS treeify at iteration {} with {} samples".format(
                self.iter + 1, self.bag.num
            )
        )
        while self.should_treeify_continue():
            node_ids = self.nodes_to_split()
            for id in node_ids:
                node = self.nodes[id]
                success, good, bad = node.try_split(
                    self.config.use_kde, self.config.min_lift
                )
                if not success:
                    continue
                good_kid = Node(self.bag, good, node)
                bad_kid = Node(self.bag, bad, node)
                node.update_kids(good_kid, bad_kid)
                self.nodes.append(good_kid)
                self.nodes.append(bad_kid)

    def select(self, at_num):
        for node in self.nodes:
            node.score = np.nan
        path = []
        current = self.root
        while not current.is_leaf():
            scores = []
            for kid in current.kids:
                scores.append(
                    kid.get_score(
                        self.config.cp, self.config.multi_beam, self.config.use_kde
                    )
                )
            choice = np.random.choice(
                np.argwhere(scores == np.max(scores)).reshape(-1), 1
            )[0]
            path.append((current, choice))
            current = current.kids[choice]
        self.selected_nodes.append(current)
        self.selects += 1
        if self.config.verbose:
            self.search_inform(at_num)
            self.select_inform(current, path)
        return [current], [path]

    def beam_select(self, at_num):
        scores = []
        nodes = self.get_leafs()
        for node in nodes:
            scores.append(
                node.get_score(
                    self.config.cp, self.config.multi_beam, self.config.use_kde
                )
            )
        for i in range(self.config.beam):
            for j in range(i + 1, len(scores)):
                if scores[i] < scores[j]:
                    scores[i], scores[j] = scores[j], scores[i]
                    nodes[i], nodes[j] = nodes[j], nodes[i]
        selected = nodes[: self.config.beam]
        self.selected_nodes.extend(selected)
        self.selects += self.config.beam
        paths = []
        for node in selected:
            path = []
            cur = node
            while cur.parent:
                direction = 0 if cur.is_good_kid() else 1
                path.append((cur.parent, direction))
                cur = cur.parent
            path = path[::-1]
            paths.append(path)
        if self.config.verbose:
            self.search_inform(at_num)
            for node, path in zip(selected, paths):
                self.select_inform(node, path)
        return selected, paths

    def simulation(self, nodes, paths, nums, func: FunctionBase):
        sampler_type = self.config.local_sampler_type
        if self.config.verbose:
            print("√ local sampling with {}".format(sampler_type))
        X = np.array([])
        idx_of_selects = []
        num_each_branch = []
        for node, path, num in zip(nodes, paths, nums):
            if not hasattr(node, "sampler"):
                node.sampler = make_sampler(self.ds, sampler_type, path)
            _X = node.sampler.draw(num)
            num_each_branch.append(len(_X))
            X = np.concatenate([X.reshape(-1, _X.shape[1]), _X], axis=0)
            if self.config.verbose:
                print("--> sampled {} points from Node{}".format(len(_X), node.id))
        new_id = list(range(self.bag.num, self.bag.num + sum(num_each_branch)))
        start = 0
        for num in num_each_branch:
            idx_of_selects.append(new_id[start : start + num])
            start += num
        Y = func.exe_batch(X)
        assert len(Y) == sum(num_each_branch)
        if self.config.verbose:
            print("√ evaluated {} points".format(len(Y)))
        if self.bag.best_sample_id >= self.bag.num - sum(num_each_branch):
            best = self.bag.best
            best = -best if self.config.inverse else best
            print("--> find new best:", best)
        return idx_of_selects

    def backpropagate(self, leafs, idx_of_selects):
        for leaf, idx in zip(leafs, idx_of_selects):
            cur = leaf
            s = "Node{}".format(cur.id)
            while cur:
                D_idx = np.concatenate([cur.D_idx, idx])
                cur.update_samples(D_idx)
                if cur is not leaf:
                    s += " -> Node{}".format(cur.id)
                cur = cur.parent
            if self.config.verbose:
                print("--> backpropagate {} samples along with {}".format(len(idx), s))

    def search(self, func: FunctionBase):
        self.bag.use_kde = self.config.use_kde
        self.metric.config.inverse = self.config.inverse
        # 保存算法配置
        self.save_config("{}/".format(func.name))
        # [0] 从已有的数据开始初始化
        if self.bag.num < self.config.init_evals:
            self.init_train(func)
            self.quick_save(func.name, force=True)
            # 评估效果及绘图
            self.quick_evaluate()
            self.quick_plot(func.name)
        # 主循环
        while self.bag.num < self.config.total_evals:
            # [1] 扩展
            self.tree_destroy()
            self.treeify()
            # [2] 执行 #selects 次选择
            cnt = 0
            while cnt < self.config.selects:
                # multi-beam 或者 naive mcts 选择
                if self.config.multi_beam:
                    leafs, paths = self.beam_select(cnt)
                else:
                    self.config.beam = 1
                    leafs, paths = self.select(cnt)
                # [3] 模拟
                num_before = self.bag.num
                idx_of_selects = self.simulation(
                    leafs, paths, [self.config.batch_evals] * self.config.beam, func
                )
                # [4] 回溯
                self.backpropagate(leafs, idx_of_selects)
                self.num_new += self.bag.num - num_before
                cnt += self.config.beam
            self.iter += 1
            # 评估效果, 绘图和保存数据
            self.quick_evaluate()
            self.quick_plot(func.name)
            self.quick_save(func.name)
        # 结束搜索
        if self.iter % self.config.dump_step != 0:
            self.quick_plot(func.name, force=True)
            self.quick_save(func.name, force=True)
        print("\nLA-MCTS search finished!\n")

    def quick_save(self, name, force=False):
        if force or (
            self.config.dump_step > 0 and self.iter % self.config.dump_step == 0
        ):
            folder_path = (
                name
                + "/LAMCTS/iter"
                + str(self.iter)
                + "_point"
                + str(self.bag.num)
                + "/"
            )
            print("")
            # self.dump(folder_path)
            # self.bag.dump(folder_path)
            self.bag.save_csv(folder_path)
            self.metric.save_csv(folder_path)
            print("")

    def save_config(self, folder):
        config = {
            "features": self.ds.features,
            "lb": self.ds.lb.tolist(),
            "ub": self.ds.ub.tolist(),
            "dpi": self.ds.dpi.tolist(),
            "nominal map": self.ds.n_map,
            "results": self.ds.results,
            "lb of results": self.ds.r_lb.tolist(),
            "ub of results": self.ds.r_ub.tolist(),
            "target": self.ds.target,
            "total evals": self.config.total_evals,
            "init evals": self.config.init_evals,
            "evals per select": self.config.batch_evals,
            "selects per treeify": self.config.selects,
            "beam width": self.config.beam,
            "multi-beam mode": self.config.multi_beam,
            "leaf size": self.config.leaf_size,
            "max depth": self.config.max_depth,
            "min lift of partition": self.config.min_lift,
            "cp": self.config.cp,
            "use kde": self.config.use_kde,
            "min to max reformulated": self.config.inverse,
            "iters per dump": self.config.dump_step,
            "verbose": self.config.verbose,
            "global sampler type": self.config.global_sampler_type,
            "local sampler type": self.config.local_sampler_type,
            "iters per metric evaluation": self.metric.config.eval_metrics_step,
            "metric dpi": self.metric.config.dpi,
            "metric regression method": self.metric.config.regression_method,
            "metric interpolate method": self.metric.config.interpolate_method,
            "ROI check rule": "{} {}".format(
                self.metric.config.IR_rule[0], self.metric.config.IR_rule[1]
            ),
            "k of KDE": self.bag.kde.config.k,
            "kernel type of KDE": self.bag.kde.config.kernel_type,
            "faiss index type of KDE": self.bag.kde.config.index_type,
            "samples per KDE update": self.bag.kde.config.update_period,
        }
        s = json.dumps(config)
        if not os.path.exists(folder):
            os.mkdir(folder)
        file = "{}config.json".format(folder)
        with open(file, "w") as f:
            f.write(s)

    def dump(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(folder + "TreeDump", "wb") as f:
            pickle.dump(self, f)
        print("[INFO] dump MCTS Tree to {}TreeDump".format(folder))

    @staticmethod
    def load(folder):
        with open(folder + "TreeDump", "rb") as f:
            mcts = pickle.load(f)
        print("[INFO] load MCTS Tree from {}TreeDump".format(folder))
        return mcts

    def select_inform(self, leaf, path):
        s = ""
        for node, direction in path:
            s += "Node{} -> ".format(node.id)
        s += "Node{}".format(leaf.id)
        print("--> select branch: " + s)

    def search_inform(self, num_select):
        print(
            "{:=^156}".format(
                "#iter {}, #select {} ~ {}".format(
                    self.iter + 1, num_select + 1, num_select + self.config.beam
                )
            )
        )
        if self.config.verbose:
            for node in self.nodes:
                if not np.isnan(node.score):
                    if len(node.kids) > 0:
                        print(
                            "Node{:<5d}: leaf? {:<1}  good? {:<1}  |  "
                            "n: {:<6d} mean: {:<8.3f}  min: {:<8.3f}  max: {:<8.3f}  |  "
                            "score: {:<8.3f}  fit_error: {:<5.2f}  good_kid: {:<5d} bad_kid: {:<5d}".format(
                                node.id,
                                node.is_leaf(),
                                node.is_good_kid(),
                                node.n,
                                node.mean,
                                node.min,
                                node.max,
                                node.score,
                                node.fit_error,
                                node.kids[0].id,
                                node.kids[1].id,
                            )
                        )
                    else:
                        print(
                            "Node{:<5d}: leaf? {:<1}  good? {:<1}  |  "
                            "n: {:<6d} mean: {:<8.3f}  min: {:<8.3f}  max: {:<8.3f}  |  "
                            "score: {:<8.3f}  fit_error: {:<5}  good_kid: {:<5} bad_kid: {:<5}".format(
                                node.id,
                                node.is_leaf(),
                                node.is_good_kid(),
                                node.n,
                                node.mean,
                                node.min,
                                node.max,
                                node.score,
                                "×",
                                "×",
                                "×",
                            )
                        )
                else:
                    if len(node.kids) > 0:
                        print(
                            "Node{:<5d}: leaf? {:<1}  good? {:<1}  |  "
                            "n: {:<6d} mean: {:<8.3f}  min: {:<8.3f}  max: {:<8.3f}  |  "
                            "score: {:<8}  fit_error: {:<5.2f}  good_kid: {:<5d} bad_kid: {:<5d}".format(
                                node.id,
                                node.is_leaf(),
                                node.is_good_kid(),
                                node.n,
                                node.mean,
                                node.min,
                                node.max,
                                "×",
                                node.fit_error,
                                node.kids[0].id,
                                node.kids[1].id,
                            )
                        )
                    else:
                        print(
                            "Node{:<5d}: leaf? {:<1}  good? {:<1}  |  "
                            "n: {:<6d} mean: {:<8.3f}  min: {:<8.3f}  max: {:<8.3f}  |  "
                            "score: {:<8}  fit_error: {:<5}  good_kid: {:<5} bad_kid: {:<5}".format(
                                node.id,
                                node.is_leaf(),
                                node.is_good_kid(),
                                node.n,
                                node.mean,
                                node.min,
                                node.max,
                                "×",
                                "×",
                                "×",
                                "×",
                            )
                        )
            print("{:=^156}".format(""))

    def draw_node(self, cur: Node, mask, values, pixels, mode, inverse, selected):
        for node in selected:
            if node.id == cur.id:
                if self.selected_mask is None:
                    self.selected_mask = np.bitwise_not(mask).astype("int")
                else:
                    self.selected_mask = self.selected_mask + np.bitwise_not(
                        mask
                    ).astype("int")
        if mode == "regression" and cur.id == 0:
            values[:, :] = cur.scissor.regressor.predict(pixels).reshape(mask.shape)
        if len(cur.kids) == 2:
            labels = cur.scissor.predict(pixels).reshape(mask.shape)
            if mode == "regression":
                if np.isnan(cur.kids[0].fit_error):
                    Good = np.ones(mask.shape) * cur.kids[0].mean
                else:
                    Good = (
                        cur.kids[0]
                        .scissor.regressor.predict(pixels)
                        .reshape(mask.shape)
                    )
                if np.isnan(cur.kids[1].fit_error):
                    Bad = np.ones(mask.shape) * cur.kids[1].mean
                else:
                    Bad = (
                        cur.kids[1]
                        .scissor.regressor.predict(pixels)
                        .reshape(mask.shape)
                    )
                Good = -Good if inverse else Good
                Bad = -Bad if inverse else Bad
            if mode == "mean":
                good_value = -cur.kids[0].mean if inverse else cur.kids[0].mean
                bad_value = -cur.kids[1].mean if inverse else cur.kids[1].mean
            elif mode == "median":
                good_value = -cur.kids[0].median if inverse else cur.kids[0].median
                bad_value = -cur.kids[1].median if inverse else cur.kids[1].median
            elif mode == "max":
                good_value = -cur.kids[0].min if inverse else cur.kids[0].max
                bad_value = -cur.kids[1].min if inverse else cur.kids[1].max
            elif mode == "min":
                good_value = -cur.kids[0].max if inverse else cur.kids[0].min
                bad_value = -cur.kids[1].max if inverse else cur.kids[1].min
            index_good = np.bitwise_and(labels == cur.scissor.good_label, mask == 0)
            index_bad = np.bitwise_and(labels == cur.scissor.bad_label, mask == 0)
            if mode == "regression":
                if cur.is_good_kid():
                    values[index_good] = Good[index_good]
                else:
                    values[index_bad] = Bad[index_bad]
            else:
                values[index_good] = good_value
                values[index_bad] = bad_value
            mask_good = np.bitwise_not(index_good)
            mask_bad = np.bitwise_not(index_bad)
            self.draw_node(
                cur.kids[0], mask_good, values, pixels, mode, inverse, selected
            )
            self.draw_node(
                cur.kids[1], mask_bad, values, pixels, mode, inverse, selected
            )

    def draw_partition(self, dpi, mode, selected_nodes, return_grids=False):
        assert len(dpi) == self.ds.dim == self.ds.cdim + self.ds.ddim
        assert mode in ["mean", "min", "max", "median", "regression"]
        cur = self.root
        lb = self.ds.lb
        ub = self.ds.ub
        G = []
        for i in range(len(dpi)):
            G.append(np.linspace(lb[i], ub[i], dpi[i]))
        GG = np.array(np.meshgrid(*G))
        GGf = np.array([x.ravel() for x in GG])
        pixels = np.transpose(GGf)
        shape = GG[0].shape
        mask = np.zeros(shape, dtype="int")
        values = np.ndarray(shape)
        values.fill(np.nan)
        self.draw_node(
            cur, mask, values, pixels, mode, self.config.inverse, selected_nodes
        )
        if return_grids:
            return GG, values
        else:
            return values

    def quick_plot(self, funcname, force=False):
        curve_step = self.plotter.config.plot_curve_step
        if force or (curve_step > 0 and self.iter % curve_step == 0):
            self.plotter.plot_curve(
                self.metric.profile,
                funcname,
                self.iter,
                self.metric.config.IR_rule,
                self.config.inverse,
            )

        part_step = self.plotter.config.plot_partition_step
        part_mode = self.plotter.config.plot_partition_mode
        if (force and self.iter > 0) or (
            part_step > 0
            and self.iter > 0
            and self.iter % part_step == 0
            and len(self.nodes) > (0 if part_mode == "regression" else 1)
        ):
            GG, Z = self.draw_partition(
                self.plotter.config.dpi, part_mode, self.selected_nodes, True
            )
            self.plotter.plot_partition(
                GG[0], GG[1], Z, funcname, self.iter, self.num_new, self.selected_mask
            )

        intp_step = self.plotter.config.plot_interpolate_step
        intp_mode = self.plotter.config.plot_interpolate_mode
        if force or (intp_step > 0 and self.iter % intp_step == 0):
            self.plotter.plot_interpolate(
                funcname, intp_mode, self.iter, inverse=self.config.inverse
            )

        dyna_step = self.plotter.config.plot_dynamics_step
        if force or (dyna_step > 0 and self.iter % dyna_step == 0):
            self.plotter.plot_dynamics(funcname, self.iter)

        density_step = self.plotter.config.plot_density_step
        if self.config.use_kde and (
            force or (density_step > 0 and self.iter % density_step == 0)
        ):
            self.plotter.plot_density(funcname, self.iter)

    def quick_evaluate(self):
        if (
            self.metric.config.eval_metrics_step > 0
            and self.iter % self.metric.config.eval_metrics_step == 0
        ):
            self.metric.evaluate(self.iter)
