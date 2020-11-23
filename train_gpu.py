"""Pretraining on GPUs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import json
import time
import numpy as np

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf

import data_utils
import model_utils
from gpu_utils import assign_to_gpu, average_grads_and_vars
import function_builder

import tensorflow as tf
from graphviz import Digraph
from tensorflow.python.client import timeline

# GPU config
flags.DEFINE_integer("num_hosts", default=1,
                     help="Number of hosts")
flags.DEFINE_integer("num_core_per_host", default=1,
                     help="Number of cores per host")
flags.DEFINE_bool("use_tpu", default=False,
                  help="Whether to use TPUs for training.")

# Experiment (data/checkpoint/directory) config
flags.DEFINE_integer("num_passes", default=1,
                     help="Number of passed used for training.")
flags.DEFINE_string("record_info_dir", default='./info',
                    help="Path to local directory containing `record_info-lm.json`.")
flags.DEFINE_string("model_dir", default='./model',
                    help="Estimator model_dir.")
flags.DEFINE_string("init_checkpoint", default=None,
                    help="checkpoint path for initializing the model.")

# Optimization config
flags.DEFINE_float("learning_rate", default=1e-4,
                   help="Maximum learning rate.")
flags.DEFINE_float("clip", default=1.0,
                   help="Gradient clipping value.")
# for cosine decay
flags.DEFINE_float("min_lr_ratio", default=0.001,
                   help="Minimum ratio learning rate.")
flags.DEFINE_integer("warmup_steps", default=0,
                     help="Number of steps for linear lr warmup.")
flags.DEFINE_float("adam_epsilon", default=1e-8,
                   help="Adam epsilon")
flags.DEFINE_string("decay_method", default="poly",
                    help="poly or cos")
flags.DEFINE_float("weight_decay", default=0.0,
                   help="weight decay")

# Training config
flags.DEFINE_integer("train_batch_size", default=16,
                     help="Size of train batch.")
flags.DEFINE_integer("train_steps", default=17,#100000,
                     help="Total number of training steps.")
flags.DEFINE_integer("iterations", default=1000,
                     help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps", default=100000,
                     help="number of steps for model checkpointing.")

# Data config
flags.DEFINE_integer('seq_len', default=0,
                     help='Sequence length for pretraining.')
flags.DEFINE_integer('reuse_len', default=0,
                     help="How many tokens to be reused in the next batch. "
                     "Could be half of seq_len")
flags.DEFINE_bool("bi_data", default=True,
                  help="Use bidirectional data streams, i.e., forward & backward.")
flags.DEFINE_integer("mask_alpha", default=6,
                     help="How many tokens to form a group.")
flags.DEFINE_integer("mask_beta", default=1,
                     help="How many tokens to mask within each group.")
flags.DEFINE_integer("num_predict", default=None,
                     help="Number of tokens to predict in partial prediction.")
flags.DEFINE_integer('perm_size', default=None,
                     help='perm size.')
flags.DEFINE_bool("uncased", False,
                  help="Use uncased inputs or not.")
flags.DEFINE_integer("n_token", 32000, help="Vocab size")

# Model config
flags.DEFINE_integer("mem_len", default=0,
                     help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False,
                  help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
                     help="Clamp length")

flags.DEFINE_integer("n_layer", default=6,
                     help="Number of layers.")
flags.DEFINE_integer("d_model", default=32,
                     help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=32,
                     help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=4,
                     help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=8,
                     help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=32,
                     help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.0,
                   help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.0,
                   help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False,
                  help="Untie r_w_bias and r_r_bias")
flags.DEFINE_string("summary_type", default="last",
                    help="Method used to summarize a sequence into a compact vector.")
flags.DEFINE_string("ff_activation", default="relu",
                    help="Activation type used in position-wise feed-forward.")
flags.DEFINE_bool("use_bfloat16", False,
                  help="Whether to use bfloat16.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
                  enum_values=["normal", "uniform"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")


FLAGS = flags.FLAGS


def profile(run_metadata, epoch=0):
    with open('profs/timeline_step' + str(epoch) + '.json', 'w') as f:
        # Create the Timeline object, and write it to a json file
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        f.write(chrome_trace)


def graph_to_dot(graph):
    dot = Digraph()
    for n in graph.as_graph_def().node:
        dot.node(n.name, label=n.name)
        for i in n.input:
            dot.edge(i, n.name)
    return dot


def get_model_fn():
    def model_fn(features, labels, mems, is_training):
        # Get loss from inputs
        total_loss, new_mems, monitor_dict = function_builder.get_loss(
            FLAGS, features, labels, mems, is_training)

        # Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        # GPU
        assert is_training
        all_vars = tf.trainable_variables()
        grads = tf.gradients(total_loss, all_vars)
        grads_and_vars = list(zip(grads, all_vars))

        return total_loss, new_mems, grads_and_vars

    return model_fn


def single_core_graph(is_training, features, mems):
    model_fn = get_model_fn()

    model_ret = model_fn(
        features=features,
        labels=None,
        mems=mems,
        is_training=is_training)

    return model_ret


def create_mems_tf(bsz_per_core):
    mems = [tf.placeholder(dtype=tf.float32,
                           shape=[FLAGS.mem_len, bsz_per_core, FLAGS.d_model])
            for layer in range(FLAGS.n_layer)]

    return mems


def initialize_mems_np(bsz_per_core):
    mems_np = [np.zeros(shape=[FLAGS.mem_len, bsz_per_core, FLAGS.d_model],
                        dtype=np.float32)
               for layer in range(FLAGS.n_layer)]

    return mems_np


def train(ps_device):
    # Get input function and model function

    train_input_fn, record_info_dict = data_utils.get_input_fn(
        tfrecord_dir=FLAGS.record_info_dir,
        split="train",
        bsz_per_host=FLAGS.train_batch_size,
        seq_len=FLAGS.seq_len,
        reuse_len=FLAGS.reuse_len,
        bi_data=FLAGS.bi_data,
        num_hosts=1,
        num_core_per_host=1,  # set to one no matter how many GPUs
        perm_size=FLAGS.perm_size,
        mask_alpha=FLAGS.mask_alpha,
        mask_beta=FLAGS.mask_beta,
        uncased=FLAGS.uncased,
        num_passes=FLAGS.num_passes,
        use_bfloat16=FLAGS.use_bfloat16,
        num_predict=FLAGS.num_predict)

    # for key, info in record_info_dict.items():
    tf.logging.info("num of batches {}".format(record_info_dict["num_batch"]))

    # Create input tensors / placeholders
    bsz_per_core = FLAGS.train_batch_size // FLAGS.num_core_per_host

    params = {
        "batch_size": FLAGS.train_batch_size  # the whole batch
    }
    train_set = train_input_fn(params)

    example = train_set.make_one_shot_iterator().get_next()

    if FLAGS.num_core_per_host > 1:
        examples = [{} for _ in range(FLAGS.num_core_per_host)]
        for key in example.keys():
            vals = tf.split(example[key], FLAGS.num_core_per_host, 0)
            for device_id in range(FLAGS.num_core_per_host):
                examples[device_id][key] = vals[device_id]
    else:
        examples = [example]

    # Create computational graph
    tower_mems, tower_losses, tower_new_mems, tower_grads_and_vars = [], [], [], []

    for i in range(FLAGS.num_core_per_host):
        reuse = True if i > 0 else None
        with tf.device(assign_to_gpu(i, ps_device)), \
                tf.variable_scope(tf.get_variable_scope(), reuse=reuse):

            # The mems for each tower is a dictionary
            mems_i = {}
            if FLAGS.mem_len:
                mems_i["mems"] = create_mems_tf(bsz_per_core)

            loss_i, new_mems_i, grads_and_vars_i = single_core_graph(
                is_training=True,
                features=examples[i],
                mems=mems_i)

            tower_mems.append(mems_i)
            tower_losses.append(loss_i)
            tower_new_mems.append(new_mems_i)
            tower_grads_and_vars.append(grads_and_vars_i)

    # average losses and gradients across towers
    if len(tower_losses) > 1:
        loss = tf.add_n(tower_losses) / len(tower_losses)
        grads_and_vars = average_grads_and_vars(tower_grads_and_vars)
    else:
        loss = tower_losses[0]
        grads_and_vars = tower_grads_and_vars[0]

    # get train op
    train_op, learning_rate, gnorm = model_utils.get_train_op(FLAGS, None,
                                                              grads_and_vars=grads_and_vars)
    global_step = tf.train.get_global_step()

    # Training loop
    # initialize mems
    tower_mems_np = []
    for i in range(FLAGS.num_core_per_host):
        mems_i_np = {}
        for key in tower_mems[i].keys():
            mems_i_np[key] = initialize_mems_np(bsz_per_core)
        tower_mems_np.append(mems_i_np)

    saver = tf.train.Saver()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.97)#allow_growth=True)

    model_utils.init_from_checkpoint(FLAGS, global_vars=True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                          gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.graph.finalize()
        run_metadata = tf.RunMetadata()
        options = tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE)

        dot_rep = graph_to_dot(tf.get_default_graph())
        # s = Source(dot_rep, filename="test.gv", format="PNG")
        with open('profs/xln.dot', 'w') as fwr:
            fwr.write(str(dot_rep))

        operations_tensors = {}
        operations_attributes = {}
        operations_names = tf.get_default_graph().get_operations()
        count1 = 0
        count2 = 0

        for operation in operations_names:
            operation_name = operation.name
            operations_info = tf.get_default_graph(
            ).get_operation_by_name(operation_name).values()

            try:
                operations_attributes[operation_name] = []
                operations_attributes[operation_name].append(
                    operation.type)
                operations_attributes[operation_name].append(tf.get_default_graph(
                ).get_tensor_by_name(operation_name + ':0').dtype._is_ref_dtype)
            except:
                pass

            if len(operations_info) > 0:
                if not (operations_info[0].shape.ndims is None):
                    operation_shape = operations_info[0].shape.as_list(
                    )
                    operation_dtype_size = operations_info[0].dtype.size
                    if not (operation_dtype_size is None):
                        operation_no_of_elements = 1
                        for dim in operation_shape:
                            if not(dim is None):
                                operation_no_of_elements = operation_no_of_elements * dim
                        total_size = operation_no_of_elements * operation_dtype_size
                        operations_tensors[operation_name] = total_size
                    else:
                        count1 = count1 + 1
                else:
                    count1 = count1 + 1
                    operations_tensors[operation_name] = -1

                #   print('no shape_1: ' + operation_name)
                #  print('no shape_2: ' + str(operations_info))
                #  operation_namee = operation_name + ':0'
                # tensor = tf.get_default_graph().get_tensor_by_name(operation_namee)
                # print('no shape_3:' + str(tf.shape(tensor)))
                # print('no shape:' + str(tensor.get_shape()))

            else:
                # print('no info :' + operation_name)
                # operation_namee = operation.name + ':0'
                count2 = count2 + 1
                operations_tensors[operation_name] = -1

                # try:
                #   tensor = tf.get_default_graph().get_tensor_by_name(operation_namee)
                # print(tensor)
                # print(tf.shape(tensor))
                # except:
                # print('no tensor: ' + operation_namee)
        print(count1)
        print(count2)

        with open('./profs/tensors_sz_32.txt', 'w') as f:
            for tensor, size in operations_tensors.items():
                f.write('"' + tensor + '"::' + str(size) + '\n')

        with open('./profs/operations_attributes.txt', 'w') as f:
            for op, attrs in operations_attributes.items():
                strr = op
                for attr in attrs:
                    strr += '::' + str(attr)
                strr += '\n'
                f.write(strr)

        fetches = [loss, tower_new_mems, global_step,
                   gnorm, learning_rate, train_op]
        iter = 0
        total_loss, prev_step = 0., -1
        while True:
            feed_dict = {}
            for i in range(FLAGS.num_core_per_host):
                for key in tower_mems_np[i].keys():
                    for m, m_np in zip(tower_mems[i][key], tower_mems_np[i][key]):
                        feed_dict[m] = m_np
            if iter % 10 == 7 or iter == 0:
                fetched = sess.run(fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
                #if iter > 0:
                profile(run_metadata, iter)
            else:
                t0 = time.time()
                fetched = sess.run(fetches, feed_dict=feed_dict)
                print(time.time() - t0)
            if iter == 0:
                mem_options = tf.profiler.ProfileOptionBuilder.time_and_memory()
                mem_options["min_bytes"] = 0
                mem_options["min_micros"] = 0
                mem_options["output"] = 'file:outfile=./profs/mem.txt'
                mem_options["select"] = ("bytes", "peak_bytes", "output_bytes",
                          "residual_bytes")
                mem = tf.profiler.profile(
                  tf.Graph(), run_meta=run_metadata, cmd="scope", options=mem_options)
                with open('profs/mem2.txt', 'w') as f:
                  f.write(str(mem))
            iter += 1

            loss_np, tower_mems_np, curr_step = fetched[:3]
            total_loss += loss_np

            if curr_step > 0 and curr_step % FLAGS.iterations == 0:
                curr_loss = total_loss / (curr_step - prev_step)
                tf.logging.info("[{}] | gnorm {:.2f} lr {:8.6f} "
                                "| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
                                    curr_step, fetched[-3], fetched[-2],
                                    curr_loss, math.exp(curr_loss), curr_loss / math.log(2)))
                total_loss, prev_step = 0., curr_step

            if curr_step > 0 and curr_step % FLAGS.save_steps == 0:
                save_path = os.path.join(FLAGS.model_dir, "model.ckpt")
                saver.save(sess, save_path)
                tf.logging.info("Model saved in path: {}".format(save_path))

            if curr_step >= FLAGS.train_steps:
                break


def main(unused_argv):
    del unused_argv  # Unused

    tf.logging.set_verbosity(tf.logging.INFO)

    # Get corpus info
    FLAGS.n_token = data_utils.VOCAB_SIZE
    tf.logging.info("n_token {}".format(FLAGS.n_token))

    if not tf.gfile.Exists(FLAGS.model_dir):
        tf.gfile.MakeDirs(FLAGS.model_dir)

    train("/gpu:0")


if __name__ == "__main__":
    tf.app.run()
