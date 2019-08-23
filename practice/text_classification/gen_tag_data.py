#!/usr/bin/env python2
#-*- coding: utf-8 -*-
#@Filename : gen_tag_data
#@Date : 2018-08-21-18-00
#@AUTHOR : bai
import tensorflow as tf
import fire
import kaka
import json
import random
MAX_TOKEN_NUM_PER_SENTENCE = 200
maxTokens = 0

def load_job_direction(direction_path):
    dir_dict = {}
    with open(direction_path, 'r') as f:
        for line in f:
            line = line.replace('\n', '')
            if not line:
                continue
            if len(line) == 0:
                continue
            todo = line.split('\t')
            if todo[0] not in dir_dict:
                dir_dict[todo[0]] = []
            dir_dict[todo[0]].append(todo[1])
    return dir_dict

def load_vocab(path):
    ret = {}
    with open(path, "r") as fp:
        for line in fp.readlines():
            line = line.strip()
            if not line:
                continue
            ss = line.split("\t")
            assert (len(ss) == 2)
            idx = int(ss[1])
            word = ss[0]
            if word == "<UNK>" or word == "</s>":
                continue
            us = word.decode("utf8")
            # assert (len(us) == 1)
            ret[us] = idx
    return ret

def gen_sentence_features(sentence, vocab, seg):
    global maxTokens
    sentence = sentence.lower()
    ss = seg.tokenize_string_ex(sentence, False)
    ss = [x for x in ss if (x != ' ' and x != '\t')]
    if len(ss) < 1:
        return None
    nt = 0
    ret = []
    for s in ss:
        s = s.strip()
        if not s:
            continue
        if s in vocab:
            idx = vocab[s]
        else:
            idx = 1
        nt += 1
        if nt <= MAX_TOKEN_NUM_PER_SENTENCE:
            ret.append(idx)
        else:
            break
    if nt > maxTokens:
        maxTokens = nt
    nt = len(ret)
    for i in range(nt, MAX_TOKEN_NUM_PER_SENTENCE):
        ret.append(0)
    return ret

def convert(data_path,
            train_output_path,
            test_output_path,
            word_vocab_path,
            job_class,
            direction_path,
            test_ratio=0.05):
    writerTrain = tf.python_io.TFRecordWriter(train_output_path)
    writerTest = tf.python_io.TFRecordWriter(test_output_path)
    seg = kaka.Tokenizer("/var/local/kakaseg/conf.json")
    vocab = load_vocab(word_vocab_path)
    job_direction_dict = load_job_direction(direction_path)
    if job_class not in job_direction_dict:
        print job_class + " not in dictionary."
        return
    all_tag = job_direction_dict[job_class]
    tag_dict = {}
    for i in range(0, len(all_tag)):
        tag_dict[all_tag[i]] = i + 1
    processed = 0
    ntrain = 0
    ntest = 0
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item['class'].encode('utf-8') == job_class:
                sf = gen_sentence_features(item['content'].encode('utf-8').lower(), vocab, seg)
                targets = []
                for t in item['tag']:
                    if t.encode('utf-8') not in tag_dict:
                        print t.encode('utf-8')
                        print line
                    idx = tag_dict[t.encode('utf-8')]
                    targets.append(idx)
                targets.sort()
                while len(targets) < 3:
                    targets.append(0)
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'target':tf.train.Feature(int64_list=tf.train.Int64List(
                    value=targets)),
                        'sentence': tf.train.Feature(int64_list=tf.train.Int64List(
                    value=sf))}
                ))
                if random.random() <= test_ratio:
                    writerTest.write(example.SerializeToString())
                    ntest += 1
                else:
                    writerTrain.write(example.SerializeToString())
                    ntrain += 1
                processed += 1
                if (processed % 100) == 0:
                    print("prcessed %d, train:%d, test:%d" % (processed, ntrain, ntest))
    writerTrain.close()
    writerTest.close()

if __name__ == '__main__':
  fire.Fire()

