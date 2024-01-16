#!usr/bin/python
# coding=utf8

from functools import wraps
import time
import numpy as np
from scipy.interpolate import griddata


def gen_pixels(ds, dpi, returnGG=False):
    """
    根据 ds 和 dpi 生成网格点，返回的网格点是展平的，使用 returnGG 控制是否返回原始结果
    """
    assert len(dpi) == ds.cdim + ds.ddim
    lb = ds.lb
    ub = ds.ub
    G = []
    for i in range(ds.cdim + ds.ddim):
        G.append(np.linspace(lb[i], ub[i], dpi[i]))
    GG = np.array(np.meshgrid(*G))
    GGf = np.array([x.ravel() for x in GG])
    pixels = np.transpose(GGf)
    if returnGG:
        return pixels, GG
    return pixels


def draw_interpolate(X, Y, pixels, method="nearest", fill_value=np.nan):
    """
    已知 X 和 Y， 获得 pixels 的插值
    """
    assert method in ["nearest", "linear", "cubic"]
    gridpoints = griddata(X, Y, pixels, method, fill_value, True)
    nans = np.isnan(gridpoints)
    if not method == "nearest":
        gridpoints[nans] = griddata(X, Y, pixels, "nearest", rescale=True)[nans]
    return gridpoints


def mytimer(function):
    """
    use decorator to time
    """

    @wraps(function)
    def function_timer(*args, **kwargs):
        print("[function: {name} start...]".format(name=function.__name__))
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print(
            "[function: {name} finished, spent time: {time:.2f}s]".format(
                name=function.__name__, time=t1 - t0
            )
        )
        return result

    return function_timer


class MyTimer(object):
    """
    timer implemented by context manager
    """

    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.name is not None:
            print(
                "[{} finished, spent time: {time:.2f}s]".format(
                    self.name, time=time.time() - self.t0
                )
            )
        else:
            print(
                "[finished, spent time: {time:.2f}s]".format(time=time.time() - self.t0)
            )
