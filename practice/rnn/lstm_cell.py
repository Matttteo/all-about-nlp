# -*- coding: utf-8 -*-
# @Author: baiyunhan
# @Date:   2019-04-29 10:04:54
# @Last Modified by:   baiyunhan
# @Last Modified time: 2019-04-29 11:56:31
import tensorflow as tf

class LSTMCell(object):
    def __init__(self,
                 num_units,
                 name,
                 initializer=None,
                 forget_bias=1.0):
        self.num_units = num_units
        self.name = name
        self.initializer = initializer
        self.forget_bias = forget_bias

    def forward(self, c, h, x):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            f_input = tf.concat([h, x], -1)
            forget_weight = tf.get_variable('forget_weight',
                                             shape=[f_input.shape.as_list()[-1], self.num_units],
                                             regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                             initializer=initializer)
            forget_bias = tf.get_variable('forget_bias',
                                          initializer=[self.forget_bias] * self.num_units)
            forget = tf.nn.xw_plus_b(f_input, forget_weight, forget_bias, name='forget')
            forget_gate = tf.math.sigmoid(forget, name="forget_gate")
            
            input_weight = tf.get_variable('input_weight',
                                             shape=[f_input.shape.as_list()[-1], self.num_units],
                                             regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                             initializer=initializer)
            input_bias = tf.get_variable('input_bias',
                                          initializer=[0.0] * self.num_units)
            input_layer = tf.nn.xw_plus_b(f_input, input_weight, input_bias, name='input')
            input_gate = tf.math.sigmoid(input_layer, name="input_gate")
            
            candidate_weight = tf.get_variable('candidate_weight',
                                             shape=[f_input.shape.as_list()[-1], self.num_units],
                                             regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                             initializer=initializer)
            candidate_bias = tf.get_variable('candidate_bias',
                                              initializer=[0.0] * self.num_units)
            candidate_layer = tf.nn.xw_plus_b(f_input, candidate_weight, candidate_bias, name='candidate')
            candidate = tf.nn.tanh(candidate_layer, name='candidate_tanh')
            
            new_candidate = tf.math.multiply(forget_gate, c) + tf.math.multiply(input_gate, candidate)
            
            output_weight = tf.get_variable('output_weight',
                                             shape=[f_input.shape.as_list()[-1], self.num_units],
                                             regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                             initializer=initializer)
            output_bias = tf.get_variable('output_bias',
                                          initializer=[0.0] * self.num_units)
            output_layer = tf.nn.xw_plus_b(f_input, output_weight, output_bias, name='output')
            output_gate = tf.math.sigmoid(output_layer, name="output_gate")
            
            new_output = tf.math.multiply(output_gate, tf.nn.tanh(new_candidate))
            
            return new_candidate, new_output
