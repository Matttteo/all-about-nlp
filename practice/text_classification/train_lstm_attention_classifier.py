# -*- coding: utf-8 -*-
# @Author: baiyunhan
# @Date:   2019-03-25 13:02:18
# @Last Modified by:   baiyunhan
# @Last Modified time: 2019-03-25 16:01:32
import tensorflow as tf
import model_lstm_attention as model


flags = tf.flags

FLAGS = flags.FLAGS


flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "train_files", None,
    "Train tfrecord file")

flags.DEFINE_string(
    "test_files", None,
    "Test tfrecord file")

flags.DEFINE_string(
    "model_config", None,
    "Model config file path.")

flags.DEFINE_string(
    "wordvec", None,
    "Pretrained wordvec file path.")

flags.DEFINE_integer(
    "feature_size", 301,
    "sentence reprsent size")

flags.DEFINE_integer(
    "num_class", 301,
    "number of classify output")

flags.DEFINE_integer("train_batch_size", 64, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 128, "Total batch size for eval.")


flags.DEFINE_float("learning_rate", 0.001, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")


flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_float(
    "warmup_proportion", 0.01,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")


def load_w2v(path):
    fp = open(path, "r")
    print("load data from:", path)
    line = fp.readline().strip()
    ss = line.split(" ")
    total = int(ss[0])
    dim = int(ss[1])
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

def create_model(model_config, wordvec, is_training, input, num_class):
    config = model.LstmAttentionConfig()
    config.from_json_file(model_config)
    word_emb = load_w2v(wordvec)
    lstm_att_model = model.ModelLstmAttention(
        model_config=model_config, input=input['feature'], wordEmd=word_emb, is_training=is_training)

    final_repr = lstm_att_model.final_repr
    alignment = lstm_att_model.retA
    with tf.variable_scope("classify", reuse=tf.AUTO_REUSE):
        final_repr_size = final_repr.shape.as_list()
        classify_out_weight = tf.get_variable(
            "classify_out_weight",
            shape=[final_repr[-1], num_class],
            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
            initializer=tf.contrib.layers.xavier_initializer())
        classify_out_bias = tf.get_variable(
            "classify_out_bias", shape=[num_titles])
        class_repr = tf.nn.xw_plus_b(
            final_repr_size,
            classify_out_weight,
            classify_out_biass,
            name="classify_out")
        loss = None
        if is_training:
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                labels=input['target'], logits=class_repr))
            loss = loss - 0.1 * tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=alignment, logits=alignment))
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss += tf.reduce_sum(reg_losses)
        return class_repr, loss

def model_fn_builder(model_config, wordvec, num_class, learning_rate,
                     num_train_steps, num_warmup_steps):

    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        if mode == tf.estimator.ModeKeys.PREDICT:
            class_repr, _ = create_model(model_config, wordvec, False, features, num_class)
            export_outputs = {
                'predict_output': tf.estimator.export.PredictOutput({
                    'class_repr': class_repr
                })
            }
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"output": class_repr},
                export_outputs=export_outputs)
            return output_spec
        class_repr, loss = create_model(model_config, wordvec, True, features, num_class)
        top3_precision = tf.metrics.average_precision_at_k(labels=features['target'], predictions=class_repr, 3)
        top5_precision = tf.metrics.average_precision_at_k(labels=features['target'], predictions=class_repr, 5)
        metrics = {
            "top3_precision" : top3_precision,
            "top5_precision" : top5_precision
        }
        if mode == tf.estimator.ModeKeys.EVAL:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=metrics)
            return output_spec
        assert (mode == tf.estimator.ModeKeys.TRAIN)
        train_op = optimization.create_optimizer(
            loss, learning_rate, num_train_steps, num_warmup_steps, False)
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=metrics,
            train_op=train_op)
        return output_spec

    return model_fn

def recviver_fn_builder(feature_size):
  # For serving
    def serving_input_receiver_fn():
        input = tf.placeholder(dtype=tf.int64, shape=[None, feature_size], name='feature')
        receiver_tensors = {'feature': input}
        features = {"feature" : input}
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
    return serving_input_receiver_fn

def file_based_input_fn_builder(input_files, is_training,
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
        d = tf.data.TFRecordDataset(input_files, compression_type="GZIP")
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
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    run_config = tf.estimator.RunConfig(
                          model_dir=FLAGS.output_dir,
                          save_checkpoints_steps=FLAGS.save_checkpoints_steps)


    model_fn = model_fn_builder(
        model_config=FLAGS.model_config,
        wordvec=FLAGS.wordvec,
        num_class=FLAGS.num_class,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

    train_files = tf.app.flags.FLAGS.train_files.split(",")
    train_input_fn = file_based_input_fn_builder(
        input_files=train_files,
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

    test_files = tf.app.flags.FLAGS.test_files.split(",")
    eval_input_fn = file_based_input_fn_builder(
        input_files=test_files,
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