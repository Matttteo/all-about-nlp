#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# @Filename : kmeans.py
# @Date : 2019-04-08-10-48
# @AUTHOR : bai

# 基础的Kmeans聚类
import numpy as np
import matplotlib.pyplot as plt


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


def data_gen_helper(values, centers, center_size, idx, center):
    if idx == values.shape[1]:
        centers.append(center[:])
        if len(centers) == center_size:
            return True
        else:
            return False
    for i in range(values.shape[0]):
        center.append(values[i][idx])
        if data_gen_helper(values, centers, center_size, idx + 1, center):
            return True
        else:
            center.pop()

# Kmeans算法是一种EM算法，对初始值敏感
def creat_init_center(dataset, cluster_size):
    max_value = np.amax(dataset, axis=0)
    min_value = np.amin(dataset, axis=0)
    values = np.array([max_value, min_value])
    num = 0
    centers = []
    center = []
    ret = []
    data_gen_helper(values, centers, cluster_size, 0, center)
    while num < cluster_size and num < len(centers):
        ret.append(centers[num])
        num += 1
    while num < cluster_size:
        todo = []
        for i in range((values.shape[1])):
            todo.append(np.random.uniform(min_value[i], max_value[i]))
        ret.append(todo)
        num += 1
    return ret


def distance(d1, d2):
    assert len(d1) == len(d2)
    d = 0.0
    for i in range(len(d1)):
        d += np.power((d1[i] - d2[i]), 2)
    return d


def estimate_center(values, idxs):
    sum_v = np.zeros([values.shape[1]])
    for i in idxs:
        sum_v += values[i]
    return sum_v / len(idxs)


def classify(centers, data):
    min_distance = 1e10
    min_idx = -1
    for i in range(len(centers)):
        d = distance(centers[i], data)
        if d < min_distance:
            min_distance = d
            min_idx = i
    return min_idx


def fit(dataset, cluster_size, max_step=100, ep=0.1):
    centers = creat_init_center(dataset, cluster_size)
    idxs = []
    for i in range(cluster_size):
        idxs.append([])
    step = 0
    while step < max_step:
        idxs = []
        for i in range(cluster_size):
            idxs.append([])
        for i in range(dataset.shape[0]):
            idxs[classify(centers, dataset[i])].append(i)
        new_centers = []
        d = 0
        for i in range(len(idxs)):
            new_centers.append(estimate_center(dataset, idxs[i]))
            d += distance(new_centers[i], centers[i])
        centers = new_centers
        step += 1
        if d < ep:
            print 'converge'
            break
    result = []
    for i in range(dataset.shape[0]):
        result.append(classify(centers, dataset[i]))
    return centers, result