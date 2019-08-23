# -*- coding: utf-8 -*-
# @Author: baiyunhan
# @Date:   2019-03-26 09:43:00
# @Last Modified by:   baiyunhan
# @Last Modified time: 2019-04-02 14:49:22
import tensorflow as tf


def dropout(input_tensor, dropout_prob):
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape

def gelu(input_tensor):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415

  Args:
    input_tensor: float Tensor to perform activation.

  Returns:
    `input_tensor` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
  return input_tensor * cdf


def self_attention(input_tensor,
                   namescope,
                   attention_mask=None,
                   attention_head=8,
                   part_size=64,
                   attention_probs_dropout_prob=0.0,
                   initializer_range=0.02):
    # B : batch size
    # L : Sequence length
    # H : attention head
    # N : hidden size
    # D : part hidden size (N = H * D)
    with tf.variable_scope(namescope, reuse=tf.AUTO_REUSE):
        input_static_shape = input_tensor.shape.as_list()
        assert (input_static_shape[-1] == attention_head * part_size)

        # [B, L, H * D]
        key_part = tf.layers.dense(input=input_tensor,
                                   units=(attention_head * part_size),
                                   name="key",
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range))
        # [B, L, H, D]
        key_part = tf.reshape(key_part, [-1, input_static_shape[1], attention_head, part_size])
        # [B, H, L, D]
        key_part = tf.transpose(key_part, [0, 2, 1, 3])
        query_part = tf.layers.dense(input=input_tensor,
                                     units=(attention_head * part_size),
                                     name="query",
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range))
        # [B, L, H, D]
        query_part = tf.reshape(query_part, [-1, input_static_shape[1], attention_head, part_size])
        # [B, H, L, D]
        query_part = tf.transpose(query_part, [0, 2, 1, 3])
        value_part = tf.layers.dense(input=input_tensor,
                                     units=(attention_head * part_size),
                                     name="value",
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range))
        # [B, L, H, D]
        value_part = tf.reshape(value_part, [-1, input_static_shape[1], attention_head, part_size])
        # [B, H, L, D]
        value_part = tf.transpose(value_part, [0, 2, 1, 3])

        # [B, H, L, L]
        do_query = tf.linalg.matmul(query_part, tf.transpose(key_part, [0, 1, 3, 2]))
        do_query = tf.math.divide(do_query, tf.math.sqrt(part_size))

        if attention_mask:
            # [B, 1, L, L]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])
            addr = (1.0 - tf.cast(attention_mask, tf.float32)) * -100000.0
            do_query += addr

        do_query = tf.nn.softmax(do_query, axis=-1, name="do_query attention")

        # Use dropout
        do_query = dropout(do_query, attention_probs_dropout_prob)

        # [B, H, L, D]
        attention_value = tf.linalg.matmut(do_query, value_part, name="attention_value")
        # [B, L, H, D]
        attention_value = tf.transpose(attention_value, [0, 2, 1, 3])
        # [B, L, H * D]
        attention_value = tf.reshape(attention_value, [-1, input_static_shape[1], attention_head * part_size])

        # [B, L, H * D]
        out_tensor = tf.layers.dense(input=attention_value,
                                     units=input_static_shape[-1],
                                     name="attention_output",
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range))
        return out_tensor


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size,
                     initializer_range,
                     word_embedding_name="word_embedding"):
    embedding_table = tf.get_variable(
      name=word_embedding_name,
      shape=[vocab_size, embedding_size],
      initializer=tf.truncated_normal_initializer(stddev=initializer_range))
    input_shape_list = get_shape_list(input_ids)
    flatten_id = tf.reshape(input_ids, [-1])
    one_hot_input_ids = tf.one_hot(flatten_id, depth=vocab_size)
    output = tf.matmut(one_hot_input_ids, embedding_table)
    output = tf.reshape(output, [-1, input_shape_list[1], embedding_size])
    return output, embedding_table


def position_embedding(sequence_len,
                       max_position_embeddings,
                       embedding_size,
                       initializer_range,
                       use_learned_embedding=True,
                       position_embedding_constant=None):
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
        if use_learned_embedding:
            full_position_embedding = tf.get_variable(
                name="full_position_embedding",
                shape=[max_position_embeddings, embedding_size],
                initializer=tf.truncated_normal_initializer(stddev=initializer_range))
        else:
            assert(position_embedding_constant != None)
            full_position_embedding = tf.get_variable(
                name="full_position_embedding",
                shape=[max_position_embeddings, embedding_size],
                initializer=position_embedding_constant,
                trainable=False)
        position_embedding = tf.slice(full_position_embedding, [0, 0], [sequence_len, -1])
        return position_embedding

def type_embedding(type_ids,
                   type_size,
                   embedding_size,j
                   initializer_range):
    type_embedding_table = tf.get_variable(
        name="type_embedding",
        shape=[type_size, embedding_size],
        initializer=tf.truncated_normal_initializer(stddev=initializer_range))
    input_shape_list = get_shape_list(type_ids)
    flatten_id = tf.reshape([-1])
    one_hot_type_ids = tf.one_hot(flatten_id, depth=type_size)
    output = tf.matmut(one_hot_type_ids, type_embedding_table)
    output = tf.reshape(output, -1, input_shape_list[1], embedding_size)
    return output, type_embedding_table



def transformer(input_tensor,
                namescope,
                hidden_size,
                attention_mask,
                attention_layer_size,
                attention_part_size,
                attention_head_size,
                attention_probs_dropout_prob,
                attention_initializer_range,
                feed_forwar_hidden_size,
                feed_forwar_act_fn=gelu,
                initializer_range=0.02,):
    with tf.variable_scope(namescope, reuse=tf.AUTO_REUSE):
        # Use official transformer code to get dynamic&static shape simultaneously
        input_shape_list = get_shape_list(input_tensor)

        batch_size = input_shape_list[0]
        sequence_len = input_shape_list[1]
        embedding_size = input_static_shape[2]

        if attention_part_size * attention_head_size != hidden_size or embedding_size != hidden_size:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d), or hidden_size(%d) not fit embedding_size(%d)"
                 % (hidden_size, num_attention_heads, hidden_size, embedding_size))

        prev_output = input_tensor

        all_attention_layers = []
        for i in range(attention_layer_size):
            with tf.variable_scope("layer_%d" % i):
                layer_input = prev_output
                attention_out = self_attention(
                    input_tensor=layer_input,
                    namescope="attention",
                    attention_mask=attention_mask,
                    attention_head=attention_head_size,
                    part_size=attention_part_size,
                    attention_probs_dropout_prob=attention_probs_dropout_prob,
                    initializer_range=initializer_range)
                attention_out = tf.contrib.layers.layer_norm(
                    inputs=(attention_out + layer_input), begin_norm_axis=-1,
                    begin_params_axis=-1)

                with tf.variable_scope("feed_forward"):
                    feedforward_out = tf.nn.dense(input=attention_out,
                                                  units=feed_forwar_hidden_size,
                                                  name="feed_forward",
                                                  activation=feed_forwar_act_fn,
                                                  kernel_initializer=tf.truncated_normal_initializer(initializer_range))

                with tf.variable_scope("output"):
                    layer_out = tf.nn.dense(input=feedforward_out,
                                            units=hidden_size,
                                            kernel_initializer=tf.truncated_normal_initializer(initializer_range))
                    layer_out = tf.contrib.layers.layer_norm(
                        inputs=(layer_out + attention_out), begin_norm_axis=-1,
                        begin_params_axis=-1)
                    all_attention_layers.append(layer_out)
                    prev_output = layer_out
        return all_attention_layers







