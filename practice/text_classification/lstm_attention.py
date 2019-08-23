#!/usr/bin/env python2
#-*- coding: utf-8 -*-
#@Filename : lstm_attention
#@Date : 2018-08-22-16-15
#@AUTHOR : bai
import tensorflow as tf
import math
import numpy as np

def parse_tfrecord_function(example_proto):
    totalTags = 54
    majorVal = 0.9
    defaultVal = 0.1 / (totalTags - 1)
    features = {
        "target":
        tf.FixedLenSequenceFeature([], tf.int64,allow_missing=True, default_value=0),
        "sentence":
        tf.FixedLenSequenceFeature(
            [], tf.int64, allow_missing=True, default_value=0)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    sentence = parsed_features["sentence"]
    sentence.set_shape([200])
    target = parsed_features["target"]
    target.set_shape([3])
    zeros = tf.cast(tf.zeros_like(target), dtype=tf.bool)
    ones = tf.cast(tf.ones_like(target), dtype=tf.bool)
    loc = tf.where(tf.not_equal(target, 0), ones, zeros)
    true_target = tf.boolean_mask(target, loc)
    print target.get_shape()
    print sentence.get_shape()
    targets = tf.sparse_to_dense(true_target, [totalTags], majorVal, defaultVal)
    return target, targets, sentence



class Model(object):
    def __init__(self,
                 maxTokenPerSetence,
                 wordsEm,
                 embeddingSize,
                 num_titles,
                 lastHiddenSize=200,
                 lstmEmSize=200,
                 jdHidden=512,
                 numPart=5):
        self.majorVal = 0.9
        self.defaultVal = 0.1 / (num_titles - 1)
        self.max_token_per_sentence = 200
        self.sentence_placeholder = tf.placeholder(
            tf.int32,
            shape=[None, self.max_token_per_sentence],
            name="sentence")
        self.label_target = tf.placeholder(tf.int32, shape=[None, 3])
        self.targets = tf.placeholder(tf.float32, shape=[None, num_titles])
        self.words = tf.Variable(wordsEm, name="words")
        self.embedding_size = embeddingSize
        self.last_hidden_size = lastHiddenSize
        self.lstm_em_size = lstmEmSize
        self.num_part = numPart
        self.Ws1 = tf.get_variable(
            "word_contex_weight",
            shape=[self.last_hidden_size, self.lstm_em_size * 2],
            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
            initializer=tf.contrib.layers.xavier_initializer())

        self.Ws2 = tf.get_variable(
            "ws2",
            shape=[self.num_part, self.last_hidden_size],
            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
            initializer=tf.contrib.layers.xavier_initializer())

        self.title_out_weight = tf.get_variable(
            "title_out_weight",
            shape=[self.lstm_em_size * 2 * self.num_part, num_titles],
            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
            initializer=tf.contrib.layers.xavier_initializer())
        self.title_out_bias = tf.get_variable(
            "title_out_bias", shape=[num_titles])
        self.learning_rate_h = tf.placeholder(tf.float32, shape=[])

    def length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def do_bi_lstm(self, X, lengths, reuse=None, dropout=False, scope="rnn_fwbw"):
        if dropout:
            X = tf.nn.dropout(X, keep_prob=0.5)
        with tf.variable_scope(scope, reuse=reuse) as scope:
            bcell = tf.nn.rnn_cell.LSTMCell(
                num_units=self.lstm_em_size, state_is_tuple=True, reuse=reuse)
            fcell = tf.nn.rnn_cell.LSTMCell(
                num_units=self.lstm_em_size, state_is_tuple=True, reuse=reuse)
            if not reuse:
                bcell = tf.nn.rnn_cell.DropoutWrapper(
                    cell=bcell, output_keep_prob=0.8)
                fcell = tf.nn.rnn_cell.DropoutWrapper(
                    cell=fcell, output_keep_prob=0.8)
            # bcell = tf.nn.rnn_cell.MultiRNNCell(
            #     cells=[bcell] * 2, state_is_tuple=True)
            # fcell = tf.nn.rnn_cell.MultiRNNCell(
            #     cells=[fcell] * 2, state_is_tuple=True)
            outputs, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                bcell,
                fcell,
                X,
                sequence_length=lengths,
                dtype=tf.float32,
                time_major=False,
                scope="biLSTM")

        return tf.concat(outputs, 2)

    def inference(self, X, reuse=None, name=None):
        dropout = True if reuse is None or not reuse else False
        xLen = self.length(X)
        seqEmds = tf.nn.embedding_lookup(self.words, X)
        X = self.do_bi_lstm(seqEmds,xLen, reuse=reuse, dropout=dropout, scope="sentence")

        # Now add attention
        # [None, 2U]
        Xr = tf.reshape(X, [-1, self.lstm_em_size * 2])

        #[D, None]
        C1 = tf.matmul(self.Ws1, tf.transpose(Xr))
        C1 = tf.nn.tanh(C1)
        if dropout:
            C1 = tf.nn.dropout(C1, 0.5)
        #[R,None]
        A = tf.matmul(self.Ws2, C1)

        A = tf.reshape(A, [self.num_part, -1, self.max_token_per_sentence])
        #[None,R,L]
        A = tf.transpose(A, [1, 0, 2])

        lengthMask = tf.cast(
            tf.sequence_mask(xLen, self.max_token_per_sentence), tf.int32)

        lengthMask = tf.cast((1 - lengthMask) * -1000000, tf.float32)

        lengthMask = tf.tile(lengthMask, [1, self.num_part])
        lengthMask = tf.reshape(
            lengthMask, [-1, self.num_part, self.max_token_per_sentence])

        A = tf.add(A, lengthMask)
        retA = tf.nn.softmax(
            A, name=None if name is None else name + "_a_sentence")

        #[None,R,2U]
        L1 = tf.matmul(retA, X)

        retH = tf.nn.xw_plus_b(
            tf.reshape(L1, [-1, self.num_part * 2 * self.lstm_em_size]),
            self.title_out_weight,
            self.title_out_bias,
            name=name)
        return retH, retA

    def train(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_h)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients = [
            None if gradient is None else tf.clip_by_norm(gradient, 5.0)
            for gradient in gradients
        ]
        train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op

    def test_loss(self):
        return self.loss(
            [self.label_target, self.targets, self.sentence_placeholder],
            reuse=True)

    def loss(self, inputs, reuse=None):
        target, targets,  X = inputs[0], inputs[1], inputs[2]
        name = None if reuse is None or not reuse else "inference_final"
        preds, alignment = self.inference(X, reuse=reuse, name=name)

        cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=targets, logits=preds))
        cost = cost - 0.1 * tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=alignment, logits=alignment))

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        cost += tf.reduce_sum(reg_losses)
        return cost