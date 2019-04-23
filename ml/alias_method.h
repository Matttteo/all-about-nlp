/*
 * @Author: baiyunhan
 * @Date:   2019-04-22 15:53:10
 * @Last Modified by:   baiyunhan
 * @Last Modified time: 2019-04-23 10:06:12
 * Copyright (c) 2018 baiyh
 * 
 * <<licensetext>>
 */
// https://en.wikipedia.org/wiki/Alias_method
// Alias Method: A efficient way to sample from a distribution

#include <vector>
#include <random>
#include <algorithm>
#include <list>

int alias_method(const std::vector<float>& probs, int sample_num, std::vector<int>& res) {
  int n = probs.size();
  if (n == 0) return -1;
  std::list<int> overfull;
  std::list<int> underfull;
  std::list<int> exactfull;

  float prob_sum = 0.0f;
  std::vector<float> post_prob(probs.size());
  for (int i = 0; i < n; ++i) {
    if (probs[i] < 0) return -1;
    prob_sum += probs[i];
    float v = static_cast<float>(n) * probs[i];
    post_prob[i] = v;
    if (v > 1.0f) {
      overfull.push_back(i);
    } else if (v == 1.0f) {
      exactfull.push_back(i);
    } else {
      underfull.push_back(i);
    }
  }
  if (prob_sum != 1.0f) return -1;

  std::vector<int> alias_table(n);
  for (int i = 0; i < n; ++i) alias_table[i] = i;
  while (!underfull.empty() && !overfull.empty()) {
    int overidx = overfull.back();
    overfull.pop_back();
    int underidx = underfull.back();
    underfull.pop_back();
    alias_table[underidx] = overidx;
    post_prob[overidx] -= (1.0f - post_prob[underidx]);
    if (post_prob[overidx] > 1.0f) {
      overfull.push_back(overidx);
    } else if (post_prob[overidx] == 1.0f) {
      exactfull.push_back(overidx);
    } else {
      underfull.push_back(overidx);
    }
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0f, 1.0f);

  for (int i = 0; i < sample_num; ++i) {
    float p = dis(gen);
    int idx = static_cast<int>(p * static_cast<float>(n));
    p = dis(gen);
    if (alias_table[idx] == idx) {
      res.push_back(idx);
    } else {
      if (post_prob[idx] > p) {
        res.push_back(idx);
      } else {
        res.push_back(alias_table[idx]);
      }
    }
  }
  return 0;
}