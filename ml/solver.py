#!/usr/bin/env python2
#-*- coding: utf-8 -*-
#@Filename : solver.py
#@Date : 2019/4/10
#@AUTHOR : bai
import numpy as np

from multi_armed_bandit import *
class EpsilonGreedySolver(object):
    def __init__(self, bandit, ep, prob=0.0):
        self.epsilon = ep
        self.bandit = bandit
        self.estimates = [prob] * bandit.n
        self.action_count = [0] * bandit.n

    def run_one_step(self):
        if np.random.uniform(0, 1) < self.epsilon:
            i = np.random.randint(0, self.bandit.n)
        else:
            i = np.argmax(self.estimates)
        r = self.bandit.get_reward(i)
        self.bandit.update_regret(r)
        self.estimates[i] += (r - self.estimates[i]) / (self.action_count[i] + 1)
        self.action_count[i] += 1

# Upper Confidence Bounds
class UCBSolver(object):
    def __init__(self, bandit, prob_th, init_prob=0.0):
        self.prob_th = prob_th
        self.bandit = bandit
        self.estimates = [init_prob] * bandit.n
        self.action_count = [0] * bandit.n
        self.const_v = np.sqrt((-1.0 * np.log(self.prob_th)) / 2)

    def run_one_step(self):
        i = np.argmax([self.estimates[j] + self.const_v / np.sqrt(1 + self.action_count[j]) for j in range(self.bandit.n)])
        r = self.bandit.get_reward(i)
        self.bandit.update_regret(r)
        self.estimates[i] += (r - self.estimates[i]) / (self.action_count[i] + 1)
        self.action_count[i] += 1

class UCBSolver2(object):
    def __init__(self, bandit, init_prob=0.0):
        self.bandit = bandit
        self.estimates = [init_prob] * bandit.n
        self.action_count = [0] * bandit.n
        self.step = 0

    def run_one_step(self):
        self.step += 1
        i = np.argmax([self.estimates[j] + np.sqrt(2 * np.log(self.step)) / np.sqrt(1 + self.action_count[j]) for j in range(self.bandit.n)])
        r = self.bandit.get_reward(i)
        self.bandit.update_regret(r)
        self.estimates[i] += (r - self.estimates[i]) / (self.action_count[i] + 1)
        self.action_count[i] += 1

class ThompsonSamplingSolver(object):
    def __init__(self, bandit, init_prob=0.0):
        self.bandit = bandit
        self.action_count = [0] * bandit.n
        self.alpha = [1.0] * bandit.n
        self.beta = [1.0] * bandit.n
        self.estimates = [init_prob] * bandit.n

    def run_one_step(self):
        samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(0, self.bandit.n)]
        i = np.argmax(samples)
        r = self.bandit.get_reward(i)
        self.bandit.update_regret(r)
        self.action_count[i] += 1
        self.alpha[i] += r
        self.beta[i] += (1 - r)
        self.estimates[i] = (self.alpha[i] / (self.alpha[i] + self.beta[i]))

def run_solver(num_step,
               bandit_arms,
               solver_name,
               bandit_prob=None,
               **kwargs):
    bandit = BernoulliBandit(bandit_arms, bandit_prob)
    solver = None
    if solver_name == 'EpsilonGreedySolver':
        solver = EpsilonGreedySolver(bandit, kwargs['ep'])
    elif solver_name == 'UCBSolver':
        solver = UCBSolver(bandit, kwargs['prob'])
    elif solver_name == 'UCBSolver2':
        solver = UCBSolver2(bandit)
    elif solver_name == 'ThompsonSamplingSolver':
        solver = ThompsonSamplingSolver(bandit)
    if solver is None:
        print "error"
        return
    for i in range(num_step):
        solver.run_one_step()
    print solver_name
    print solver.estimates
    print bandit.probs
    print solver.action_count
    print bandit.regret_accumulate

if __name__ == '__main__':
    bandit_prob = [np.random.uniform(0, 1) for _ in range(10)]
    run_solver(1000, 10, "EpsilonGreedySolver", bandit_prob, ep=0.1)
    run_solver(1000, 10, "UCBSolver", bandit_prob, prob=0.1)
    run_solver(1000, 10, "UCBSolver2", bandit_prob)
    run_solver(1000, 10, "ThompsonSamplingSolver", bandit_prob)