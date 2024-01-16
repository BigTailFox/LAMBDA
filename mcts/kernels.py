#!usr/bin/python
# coding=utf8
import numpy as np


def gaussian(X, dim):
    """
    协方差矩阵为单位阵的高斯核函数，由于不需要归一化，省略了常系数项
    """
    return np.power(2.0 * np.pi, -0.5 * dim) * np.exp(-0.5 * np.power(X, 2))


# def log_gaussian(X, dim):
#     '''
#     协方差矩阵为单位阵的高斯核函数，由于不需要归一化，省略了常系数项
#     注意这个函数返回对数似然
#     '''
#     return -0.5 * (np.power(X, 2) + dim * np.log(2.0 * np.pi))


def make_kernel(type):
    if type == "gaussian":
        return gaussian
    # elif type == 'log gaussian':
    #     return log_gaussian
    else:
        raise RuntimeError("kde kernel {} not implemented".format(type))


# X = np.array([0,1,2,3,4,5])
# print(gaussian(X))
# print(log_gaussian(X))
# print(np.log(gaussian(X)))
