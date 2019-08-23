# -*- coding: utf-8 -*-
# @Author: baiyunhan
# @Date:   2019-07-22 11:33:55
# @Last Modified by:   baiyunhan
# @Last Modified time: 2019-07-25 17:59:44
import tensorflow as tf
from tensorflow.python.ops import array_ops

def focal_loss(input_tensor, target_tensor, alpha, gamma):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     input_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    preds = tf.nn.sigmoid(input_tensor)
    zeros = array_ops.zeros_like(preds, dtype=preds.dtype)
    ones = array_ops.ones_like(preds, dtype=preds.dtype)
    neg_pred = ones - preds

    zeros_active = tf.equal(target_tensor, zeros)
    zeros_active = tf.cast(zeros_active, tf.float32)
    neg_part = - zeros_active * (1.0 - alpha) * tf.pow(preds, gamma) * tf.log(tf.clip_by_value(neg_pred, 1e-8, 1.0))

    ones_active = tf.equal(target_tensor, ones)
    ones_active = tf.cast(ones_active, tf.float32)
    pos_part = - ones_active * alpha * tf.pow(neg_pred, gamma) * tf.log(tf.clip_by_value(preds, 1e-8, 1.0))

    loss = pos_part + neg_part
    loss = tf.reduce_sum(loss, 1)
    loss = tf.reduce_mean(loss)
    return loss