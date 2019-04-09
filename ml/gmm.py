#!/usr/bin/env python2
#-*- coding: utf-8 -*-
#@Filename : gmm.py
#@Date : 2019-04-08-10-48
#@AUTHOR : bai

# 使用EM算法实现对高斯混合模型的最大似然参数估计
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def gen_one_data(data_size,
                 hidden_mean,
                 hidden_var):
    assert len(hidden_mean) == len(hidden_var), "hidden size invalid"
    cov = np.diag(hidden_var)
    dataset = np.random.multivariate_normal(hidden_mean, cov, data_size)
    return dataset

def gen_dataset(alpha,
                total_size,
                means,
                vars,
                show=True):
    datasets = []
    colors = []
    for i in range(0, len(alpha)):
        a = alpha[i]
        part_size = int(total_size * a)
        data = gen_one_data(part_size, means[i], vars[i])
        for j in range(0, data.shape[0]):
            colors.append((i + 1) / 10.0)
        datasets.append(data)
    dataset = np.concatenate(datasets)
    if show:
        x, y = dataset.T
        plt.scatter(x, y, c=colors, alpha=0.5)
        plt.show()
    return dataset

def estimate_gamma(dataset, alpha, means, vars, gamma):
    num, part = gamma.shape
    for i in range(0, num):
        todo = []
        for j in range(0, part):
            a = alpha[j]
            mean = means[j]
            var = vars[j]
            x = a * multivariate_normal.pdf(dataset[i], mean, var)
            todo.append(x)
        sum = 0.0
        for t in todo:
            sum += t
        for j in range(0, part):
            gamma[i][j] = todo[j] / sum

def calc_log_likehood(dataset, alpha, means, vars):
    num = dataset.shape[0]
    part = len(alpha)
    log_likehood = 0.0
    for i in range(0, num):
        x = 0.0
        for j in range(0, part):
            a = alpha[j]
            mean = means[j]
            var = vars[j]
            x += a * multivariate_normal.pdf(dataset[i], mean, var)
        x = np.log(x)
        log_likehood += x
    return log_likehood

def fit(dataset,
        init_alpha,
        init_means,
        init_vars,
        max_step=100,
        ep=0.1):
    part_num = len(init_alpha)
    data_num = dataset.shape[0]
    gamma = np.zeros([dataset.shape[0], part_num])
    covs = []
    for var in init_vars:
        covs.append(np.diag(var))
    alpha = init_alpha
    means = init_means
    estimate_gamma(dataset, alpha, means, covs, gamma)
    print gamma[0]
    loglikehood = calc_log_likehood(dataset, alpha, means, covs)
    print loglikehood
    for i in range(0, max_step):
        for j in range(0, part_num):
            gamma_sigma = 0.0
            for k in range(0, data_num):
                gamma_sigma += gamma[k, j]
            gamma_data_sigma = np.zeros([2])
            cov = np.zeros((2, 2))
            for k in range(0, data_num):
                gamma_data_sigma += gamma[k][j] * dataset[k]
                delta = dataset[k] - means[j]
                cov += np.dot(delta.reshape(2, 1), delta.reshape(1, 2)) * gamma[k][j]
            means[j] = gamma_data_sigma / gamma_sigma
            covs[j] = cov / gamma_sigma
            alpha[j] = gamma_sigma / data_num
        new_loglikehood = calc_log_likehood(dataset, alpha, means, covs)
        print new_loglikehood
        if new_loglikehood - loglikehood < ep:
            print 'converge'
            break
        loglikehood = new_loglikehood
        estimate_gamma(dataset, alpha, means, covs, gamma)
    return alpha, means, covs


def inference(dataset, colors, alpha, means, vars):
    part_num = len(alpha)
    data_num = dataset.shape[0]
    color_list = ['r', 'g', 'b', 'y', 'k']
    for i in range(0, data_num):
        max_part = -1
        max_p = -1.0
        for j in range(0, part_num):
            a = alpha[j]
            mean = means[j]
            var = vars[j]
            x = a * multivariate_normal.pdf(dataset[i], mean, var)
            if x > max_p:
                max_p = x
                max_part = j
        colors.append(color_list[max_part])
    return colors

# EM 算法对初始值很敏感