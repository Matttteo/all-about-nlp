#!/usr/bin/env python2
#-*- coding: utf-8 -*-
#@Filename : train_lstm_att
#@Date : 2018-08-22-16-25
#@AUTHOR : bai
import os
import math

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
import lstm_attention as direction_model


tf.app.flags.DEFINE_string('train_data_path', "/Users/higgs/codebase/all-about-nlp/practice/text_classification/train_yy.tfrecord",
                           'Training data path')
tf.app.flags.DEFINE_string('test_data_path', "/Users/higgs/codebase/all-about-nlp/practice/text_classification/test_yy.tfrecord",
                           'Test data path')
tf.app.flags.DEFINE_string('log_dir', "logs_sa_2", 'The log  dir')
tf.app.flags.DEFINE_string("wordvec_path", "/Users/higgs/PycharmProjects/recommend/data/resume_wvec.txt",
                           "the word word2vec data path")
tf.app.flags.DEFINE_integer("wordvec_size", 150, "the vec embedding size")
tf.app.flags.DEFINE_integer("max_tokens_per_sentence", 200,
                            "max num of tokens per sentence")

tf.app.flags.DEFINE_integer("num_titles", 53, "number of title")
tf.app.flags.DEFINE_integer("max_epochs", 100, "max num of epoches")

tf.app.flags.DEFINE_integer("batch_size", 128, "num example per mini batch")
tf.app.flags.DEFINE_integer("test_batch_size", 256,
                            "num example per test batch")
tf.app.flags.DEFINE_integer("train_steps", 380000, "trainning steps")
tf.app.flags.DEFINE_integer("track_history", 15, "track max history accuracy")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_float("learning_rate_min", 0.00001,
                          "the final minimal learning rate")

def load_w2v(path, expectDim):
    fp = open(path, "r")
    print("load data from:", path)
    line = fp.readline().strip()
    ss = line.split(" ")
    total = int(ss[0])
    dim = int(ss[1])
    assert (dim == expectDim)
    ws = []
    mv = [0 for i in range(dim)]
    second = -1
    for t in range(total):
        if ss[0] == '<UNK>':
            second = t
        line = fp.readline().strip()
        ss = line.split(" ")
        assert (len(ss) == (dim + 1))
        vals = []
        for i in range(1, dim + 1):
            fv = float(ss[i])
            mv[i - 1] += fv
            vals.append(fv)
        ws.append(vals)
    for i in range(dim):
        mv[i] = mv[i] / total
    assert (second != -1)
    # append two more token , maybe useless
    ws.append(mv)
    ws.append(mv)
    if second != 1:
        t = ws[1]
        ws[1] = ws[second]
        ws[second] = t
    fp.close()
    return np.asarray(ws, dtype=np.float32)


def test_eval(sess, loss, testDatas, batchSize, sentence_h,
              target_h, targets_h):
    totalLen = len(testDatas)
    numBatch = int((totalLen - 1) / batchSize) + 1
    totalLoss = 0
    for i in range(numBatch):
        endOff = (i + 1) * batchSize
        if endOff > totalLen:
            endOff = totalLen
        X = testDatas[i * batchSize:endOff]
        target = [x[0] for x in X]
        targets = [x[1] for x in X]
        sentences = [x[2] for x in X]
        feed_dict = {sentence_h: sentences,
                     target_h: target,
                     targets_h: targets}
        lossv = sess.run(loss, feed_dict)
        totalLoss += lossv
    return totalLoss / numBatch


def load_test_dataset_all(sess, test_input, testDatas):
    while True:
        try:
            target, targets, sentence = sess.run(test_input)
            testDatas.append((target, targets, sentence))
        except tf.errors.OutOfRangeError:
            break


def main(unused_argv):
    curdir = os.path.dirname(os.path.realpath(__file__))
    trainDataPaths = tf.app.flags.FLAGS.train_data_path.split(",")
    for i in range(len(trainDataPaths)):
        if not trainDataPaths[i].startswith("/"):
            trainDataPaths[i] = curdir + "/../../" + trainDataPaths[i]

    testDataPaths = tf.app.flags.FLAGS.test_data_path.split(",")
    for i in range(len(testDataPaths)):
        if not testDataPaths[i].startswith("/"):
            testDataPaths[i] = curdir + "/../../" + testDataPaths[i]
    graph = tf.Graph()
    testDatas = []
    with graph.as_default():
        datasetTrain = tf.data.TFRecordDataset(trainDataPaths)
        datasetTrain = datasetTrain.map(direction_model.parse_tfrecord_function)
        datasetTrain = datasetTrain.repeat(FLAGS.max_epochs)
        datasetTrain = datasetTrain.shuffle(buffer_size=20000)
        datasetTrain = datasetTrain.batch(FLAGS.batch_size)
        iterator = datasetTrain.make_one_shot_iterator()
        batch_inputs = iterator.get_next()
        print("batch shape:%r" % (batch_inputs[2].get_shape()))
        datasetTest = tf.data.TFRecordDataset(testDataPaths)
        datasetTest = datasetTest.map(direction_model.parse_tfrecord_function)

        iteratorTest = datasetTest.make_initializable_iterator()
        test_input = iteratorTest.get_next()
        wordsEm = load_w2v(FLAGS.wordvec_path, FLAGS.wordvec_size)
        model = direction_model.Model(FLAGS.max_tokens_per_sentence, wordsEm,
                             FLAGS.wordvec_size, FLAGS.num_titles + 1)
        print("train data path:", trainDataPaths)
        loss = model.loss(batch_inputs)
        testLoss= model.test_loss()
        train_op = model.train(loss)
        decayPerStep = (
            FLAGS.learning_rate - FLAGS.learning_rate_min) / FLAGS.train_steps
        sv = tf.train.Supervisor(graph=graph, logdir=FLAGS.log_dir)
        with sv.managed_session(master='') as sess:
            # actual training loop
            training_steps = FLAGS.train_steps
            bestLoss = float("inf")
            trackHist = 0
            sess.run(iteratorTest.initializer)
            load_test_dataset_all(sess, test_input, testDatas)
            tf.train.write_graph(sess.graph_def,
                                 FLAGS.log_dir,
                                 "graph.pb",
                                 as_text=False)
            print("Loaded #tests:%d" % (len(testDatas)))
            for step in range(training_steps):
                if sv.should_stop():
                    break
                try:
                    clipStep = int(step / 20000)
                    clipStep = clipStep * 20000
                    trainLoss, _ = sess.run(
                        [loss, train_op], {model.learning_rate_h: (
                            FLAGS.learning_rate - decayPerStep * clipStep)})
                    if (step + 1) % 100 == 0:
                        print("[%d] loss: [%r]" % (step + 1, trainLoss))
                    if (step + 1) % 200 == 0 or step == 0:
                        tloss= test_eval(
                            sess, testLoss, testDatas,
                            FLAGS.test_batch_size, model.sentence_placeholder,
                            model.label_target, model.targets)

                        print("test loss:%.3f" %
                              (tloss))
                        if step and tloss < bestLoss:
                            sv.saver.save(sess, FLAGS.log_dir + '/best_model')
                            trackHist = 0
                            bestLoss = tloss
                        else:
                            if trackHist >= FLAGS.track_history:
                                print(
                                    "always not good enough in last %d histories, best accuracy:%.3f"
                                    % (trackHist, bestLoss))
                                break
                            else:
                                trackHist += 1
                except KeyboardInterrupt, e:
                    sv.saver.save(sess,
                                  FLAGS.log_dir + '/model',
                                  global_step=(step + 1))
                    raise e
            sv.saver.save(sess, FLAGS.log_dir + '/finnal-model')


if __name__ == '__main__':
    tf.app.run()