# -*- coding: utf-8 -*-
# @Author: baiyunhan
# @Date:   2019-03-25 10:26:14
# @Last Modified by:   baiyunhan
# @Last Modified time: 2019-03-25 13:41:22

import tensorflow as tf

class LstmAttentionConfig(object):
  def __init__(self,
               hidden_size=200,
               attention_part=5,
               lstm_emb_size=200,
               drop_out=0.0):
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.lstm_emb_size = lstm_emb_size
    self.drop_out = drop_out

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `LstmAttentionConfig` from a Python dictionary of parameters."""
    config = LstmAttentionConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `LstmAttentionConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class ModelLstmAttention(object):
    def __init__(self, model_config, input, wordEmd, is_trainning):
        self.hidden_size = model_config.hidden_size
        self.attention_part = model_config.attention_part
        self.lstm_emb_size = model_config.lstm_emb_size
        self.drop_out = model_config.drop_out
        self.word_em = tf.Variable(wordEmd, name="word_embedding")
        self.final_repr = inference(input, is_trainning)

    def inference(self, input, is_trainning):
        drop_out = self.drop_out if is_trainning else None
        self.seq_emds = tf.nn.embedding_lookup(self.word_em, input)

        if drop_out not None:
            seq_emds = tf.nn.dropout(seq_emds, keep_prob=drop_out)

        with tf.variable_scope("bilstm", reuse=tf.AUTO_REUSE):
            lstm_layer = tf.keras.layers.LSTM(
                    units=self.lstm_emb_size, dropout=drop_out)
            self.lstm_repr = tf.keras.layers.Bidirectional(lstm_layer)(seq_emds)

        attention_repr = structured_attention(
            self.lstm_repr, self.hidden_size, self.attention_part, drop_out, "structured_attention")
        return attention_repr

def length(data):
    used = tf.sign(tf.abs(data))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def structured_attention(input, hidden_size, num_part, drop_out, namescope):
    with tf.variable_scope(namescope, reuse=tf.AUTO_REUSE):
        # [batch_size, sentence_max_len, 2 * lstm_emb_size]
        input_static_shape = input.shape.as_list()
        input_length = length(input)
        input_r = tf.reshape(input, [-1, input_static_shape[-1]])

        Ws1 = tf.get_variable(
            "word_contex_weight",
            shape=[hidden_size, input_static_shape[-1]],
            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
            initializer=tf.contrib.layers.xavier_initializer())

        # [hidden_size, batch_size * sentence_max_len]
        C1 = tf.matmul(Ws1, tf.transpose(input_r))
        C1 = tf.nn.tanh(C1)

        if drop_out:
            C1 = tf.nn.drop_out(C1, drop_out)

        Ws2 = tf.get_variable(
            "ws2",
            shape=[num_part, hidden_size],
            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
            initializer=tf.contrib.layers.xavier_initializer())

        # [num_part, batch_size * sentence_max_len]
        A = tf.matmul(Ws2, C1)

        # [num_part, batch_size, sentence_max_len]
        A = tf.reshape(A, [num_part, -1, input_static_shape[1]])
        # [batch_size, num_part, sentence_max_len]
        A = tf.transpose(A, [1, 0, 2])

        # Mask padding token out
        lengthMask = tf.cast(
            tf.sequence_mask(input_length, input_static_shape[1]), tf.int32)

        lengthMask = tf.cast((1 - lengthMask) * -1000000, tf.float32)

        lengthMask = tf.tile(lengthMask, [1, self.num_part])
        # [batch_size, num_part, sentence_max_len]
        lengthMask = tf.reshape(
            lengthMask, [-1, self.num_part, self.max_token_per_sentence])

        A = tf.add(A, lengthMask)
        retA = tf.nn.softmax(
            A, name=None if name is None else name + "_a_sentence")
        self.retA = retA

        # [batch_size, num_part, 2 * lstm_emb_size]
        L1 = tf.matmul(retA, input)
        # [batch_size, num_part * 2 * lstm_emb_size]
        L1 = tf.reshape(L1, [-1, num_part * input_static_shape[-1]])
        return L1

