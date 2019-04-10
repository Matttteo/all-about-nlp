#!/usr/bin/env python2
#-*- coding: utf-8 -*-
#@Filename : multi_armed_bandit.py
#@Date : 2019/4/10
#@AUTHOR : bai
import numpy as np


class Bandit(object):
    def get_reward(self, action_id):
        raise NotImplementedError()

    def update_regret(self, action_id):
        raise NotImplemented()


class BernoulliBandit(Bandit):
    def __init__(self, size, prob=None):
        if prob is not None:
            assert len(prob) == size
            self.probs = prob
        else:
            self.probs = [np.random.uniform(0, 1) for _ in range(size)]
        self.n = size
        self.regret_accumulate = 0.0
        self.regrets = [0.0]

    def get_reward(self, action_id):
        if np.random.uniform(0, 1) < self.probs[action_id]:
            return 1
        else:
            return 0

    def update_regret(self, reward):
        regret = np.max(self.probs) - reward
        self.regret_accumulate += np.max(self.probs) - reward
        self.regrets.append(np.max(self.probs) - reward)