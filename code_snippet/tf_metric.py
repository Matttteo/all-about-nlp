# -*- coding: utf-8 -*-
# @Author: baiyunhan
# @Date:   2019-06-10 09:27:56
# @Last Modified by:   baiyunhan
# @Last Modified time: 2019-06-10 11:04:12

import tensorflow as tf

class ModelMetric(object):
    def __init__(self):
        self.metric_vars_initializer = None
        self.update_ops = {}
        self.metrics = {}

    def do_update(self, sess, extra_target, feeds):
        extra_target_size = len(extra_target)
        update_ops_list = []
        for key in update_ops:
            update_ops_list.append(update_ops[key])
        run_ops = extra_target + update_ops_list
        run_res = sess.run(run_ops, feeds)
        return run_res[:extra_target_size]

    def reset(self, sess):
        sess.run(sess.metric_vars_initializer)

    def get_model_metrics(self, sess):
        metrics_targets = []
        metrics_targets_names = []
        for key in metrics:
            metrics_targets.append(metrics[key])
            metrics_targets_names.append(key)

        metrics_res = sess.run(metrics_targets)

        metrics_dict = {}
        for i in range(0, len(metrics_targets)):
            metrics_dict[metrics_targets_names[i]] = metrics_res[i]
        return metrics_dict


    def make_metric(self, target, preds):
        p1_metric, p1_metric_update = tf.metrics.precision_at_k(m.target,
                                                                m.preds,
                                                                1,
                                                                name="precision1")
        self.update_ops['precision1'] = p1_metric_update
        self.metrics["precision1"] = p1_metric
        p3_metric, p3_metric_update = tf.metrics.precision_at_k(m.target,
                                                                m.preds,
                                                                3,
                                                                name="precision3")
        self.update_ops['precision3'] = p3_metric_update
        self.metrics["precision3"] = p3_metric
        p5_metric, p5_metric_update = tf.metrics.precision_at_k(m.target,
                                                                m.preds,
                                                                5,
                                                                name="precision5")
        self.update_ops['precision5'] = p5_metric_update
        self.metrics["precision5"] = p5_metric
        metric_val_list = []
        metric_val_list.extend(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="precision1"))
        metric_val_list.extend(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="precision3"))
        metric_val_list.extend(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="precision5"))
        self.metric_vars_initializer = tf.variables_initializer(var_list=metric_val_list)


# Use case
def make_feed_dict(model,
                   one_batch_feature_map,
                   droprate=0):
    dicts = {model.target: one_batch_feature_map["label_target"],
             model.targets: one_batch_feature_map["targets"],
             model.sentence_holder: one_batch_feature_map["sentence"],
             model.check_alen_holder: one_batch_feature_map["a_length"],
             model.check_tlen_holder: one_batch_feature_map["ab_length"],
             model.dropout_h: droprate}
    return dicts


def test_eval(sess, model, loss, test_input_feature_map, model_metrics):
    numBatch = 0
    totalLoss = 0
    model_metrics.reset(sess)
    while True:
        try:
            one_batch_feature_map = sess.run(test_input_feature_map)
            feed_dict = make_feed_dict(model, one_batch_feature_map)
            lossv = model_metrics.do_update(sess, [loss], feed_dict)
            totalLoss += lossv[0]
            numBatch += 1
            if numBatch == 20:
                break
        except tf.errors.OutOfRangeError:
            break
    totalLoss /= numBatch
    metrics = model_metrics.get_model_metrics(sess)
    print "test loss: " : totalLoss
    for key in metrics:
        print key + ": "  + str(metrics[key])
    return totalLoss
