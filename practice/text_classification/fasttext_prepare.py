#!/usr/bin/env python2
#-*- coding: utf-8 -*-
#@Filename : fasttext_prepare.py
#@Date : 2018-08-23-14-32
#@AUTHOR : bai
import kaka
import fire
import json
import random
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
    ss = [x for x in ss if (x != ' ' and x != '\t' and x != '\n')]
    return ' '.join(ss)

def convert(data_path,
            train_output_path,
            test_output_path,
            word_vocab_path,
            job_class,
            direction_path,
            test_ratio=0.05):
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
    train_lst = []
    test_lst = []
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
                example = '__label__' + str(targets[0]) + ' , ' + sf

                if random.random() <= test_ratio:
                    test_lst.append(example.encode('utf-8'))
                    ntest += 1
                else:
                    train_lst.append(example.encode('utf-8'))
                    ntrain += 1
                processed += 1
                if (processed % 100) == 0:
                    print("prcessed %d, train:%d, test:%d" % (processed, ntrain, ntest))

    with open(train_output_path, 'w') as f:
        for ex in train_lst:
            f.write(ex + '\n')
    with open(test_output_path, 'w') as f:
        for ex in test_lst:
            f.write(ex + '\n')

if __name__ == '__main__':
  fire.Fire()

