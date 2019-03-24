# -*- coding: utf-8 -*-
# @Author: baiyunhan
# @Date:   2019-03-22 23:18:32
# @Last Modified by:   bai
# @Last Modified time: 2019-03-24 20:33:15

import tensorflow as tf
import model as model


flags = tf.flags

FLAGS = flags.FLAGS


flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "train_file", None,
    "Train tfrecord file")

flags.DEFINE_string(
    "test_file", None,
    "Test tfrecord file")

flags.DEFINE_integer(
    "feature_size", 301,
    "sentence reprsent size")

flags.DEFINE_integer("train_batch_size", 64, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 128, "Total batch size for eval.")


flags.DEFINE_float("learning_rate", 0.001, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")


flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")



def model_fn_builder(feature_size,learning_rate):
	model_config = {}
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        tf.logging.info("*** Features ***")
        feature = features['feature']
        model_example = model.ModelExample(model_config=model_config, input=feature)
        if mode == tf.estimator.ModeKeys.PREDICT:
        	# 这是单个output的情况，如果有多个，其中一个的key必须是signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY.
            export_outputs = {
                'predict_output': tf.estimator.export.PredictOutput({
                    'output': model_example.output
                })
            }
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"output": model_example.output},
                export_outputs=export_outputs)
            return output_spec

        target_value = features['target']
        # Some loss
        loss = tf.losses.mean_squared_error(model_example.output, target_value)
        # 这里可以有一些metric，比如ACC等
        if mode == tf.estimator.ModeKeys.EVAL:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss)
            return output_spec
        assert (mode == tf.estimator.ModeKeys.TRAIN)
        # Some optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        global_step = tf.train.get_global_step()
        update_global_step = tf.assign(global_step, global_step + 1, name='update_global_step')
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=tf.group(train_op, update_global_step))
        return output_spec


    return model_fn

def recviver_fn_builder(feature_size):
	# For serving
    def serving_input_receiver_fn():
        input = tf.placeholder(dtype=tf.int64, shape=[None, feature_size], name='feature')
        # 这里可以有一些数据预处理，建议和训练时的预处理复用代码，避免错误
        receiver_tensors = {'feature': input}
        features = {"feature" : input}
        # receiver_tensors 是在signature部分
        # features 是喂给模型(也就是model_fn第一个参数)的
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
    return serving_input_receiver_fn

def file_based_input_fn_builder(input_file, is_training,
                                drop_remainder,  feature_size, batch_size):
    name_to_features = {}
  
    name_to_features['feature'] = tf.FixedLenFeature([feature_size], tf.int64)
    name_to_features['target'] = tf.FixedLenFeature([], tf.int64)
  
    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
  
        return example
  
    def input_fn(params):
        """The actual input function."""
    
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file, compression_type="GZIP")
        if is_training:
          d = d.repeat()
          d = d.shuffle(buffer_size=100)
    
        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
  
        return d

    return input_fn

# 获取数据集大小
def get_tf_record_size(file_path):
    options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.GZIP)
    c = 0
    for _ in tf.python_io.tf_record_iterator(file_path, options=options):
        c += 1
    return c

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    num_train_examples = None
    num_train_steps = None
    num_train_examples = get_tf_record_size(FLAGS.train_file)
    num_train_steps = int(
          (num_train_examples / FLAGS.train_batch_size) * FLAGS.num_train_epochs)
  
    run_config = tf.estimator.RunConfig(
                          model_dir=FLAGS.output_dir,
                          save_checkpoints_steps=FLAGS.save_checkpoints_steps)


    model_fn = model_fn_builder(
        feature_size=FLAGS.feature_size,
        learning_rate=FLAGS.learning_rate)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

    train_input_fn = file_based_input_fn_builder(
        input_file=FLAGS.train_file,
        is_training=True,
        drop_remainder=True,
        feature_size=FLAGS.feature_size,
        batch_size=FLAGS.train_batch_size)

    # Early stoping
    early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
        estimator,
        metric_name='loss',
        max_steps_without_decrease=1000,
        min_steps=100)
    train_spec = tf.estimator.TrainSpec(train_input_fn, hooks=[early_stopping], max_steps=num_train_steps)

    eval_input_fn = file_based_input_fn_builder(
        input_file=FLAGS.test_file,
        is_training=False,
        drop_remainder=True,
        feature_size=FLAGS.feature_size,
        batch_size=FLAGS.eval_batch_size)

    # Save best model
    exporter = tf.estimator.BestExporter(
      name="best_exporter",
      serving_input_receiver_fn=recviver_fn_builder(FLAGS.feature_size),
      exports_to_keep=5)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=1000,
                                      exporters=exporter,
                                      throttle_secs=20)

    result = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    tf.app.run()