# -*- coding: utf-8 -*-
# @Author: baiyunhan
# @Date:   2019-03-22 23:11:15
# @Last Modified by:   bai
# @Last Modified time: 2019-03-24 20:21:41

import tensorflow as tf

class ModelExample(object):
	def __init__(self, model_config, input):
		with tf.variable_scope('variables', reuse=tf.AUTO_REUSE):
			self.weight = tf.get_variable("weight", shape=[model_config.weight_size], dtype=tf.float32)
		self.output = self.inference(input)

	def inference(self, input):
		with tf.variable_scope('inference', reuse=tf.AUTO_REUSE):
			return input
