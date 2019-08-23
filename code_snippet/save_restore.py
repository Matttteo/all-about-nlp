# -*- coding: utf-8 -*-
# @Author: baiyunhan
# @Date:   2019-06-10 13:41:03
# @Last Modified by:   baiyunhan
# @Last Modified time: 2019-06-10 13:49:44
import tensorflow as tf


# import meta graph

saver = tf.train.import_meta_graph('./logs_industry_cls/best_model.meta') 

# restore variables
saver.restore(sess, 'model.ckpt')

# Get all nodes
nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]


# get tensor by name

tf.get_default_graph().get_tensor_by_name('Const:0')


# freeze graph
com = 'python freeze_graph.py  --input_graph=./logs_industry_cls/graph.pbtxt'
      '--input_checkpoint=./logs_industry_cls/best_model'
      '--output_graph=./freezed_model.pb --output_node_names=xw_plus_b_1'

com2 = 'python freeze_graph.py  --input_graph=./logs_industry_cls/graph.pb'
      '--input_checkpoint=./logs_industry_cls/best_model'
      '--output_graph=./freezed_model.pb --output_node_names=xw_plus_b_1'
      '--input_binary'