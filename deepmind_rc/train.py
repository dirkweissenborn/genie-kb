import os
import time
from kb import KB
from deepmind_rc.sampler import *
from deepmind_rc import eval
from model.models import QAModel
import tensorflow as tf
import sys
import shutil
import json
import functools
import numpy as np

# data loading specifics
tf.app.flags.DEFINE_string('kb', None, 'Path to prepared RC KB.')

# model
tf.app.flags.DEFINE_integer("size", 256, "hidden size of model")

# training
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay", 0.9, "Learning rate decay when loss on validation set does not improve.")
tf.app.flags.DEFINE_integer("batch_size", 25, "Number of examples in each batch for training.")
tf.app.flags.DEFINE_integer("max_iterations", -1, "Maximum number of batches during training. -1 means until convergence")
tf.app.flags.DEFINE_integer("ckpt_its", 1000, "Number of iterations until running checkpoint. Negative means after every epoch.")
tf.app.flags.DEFINE_integer("random_seed", 1234, "Seed for rng.")
tf.app.flags.DEFINE_integer("subsample_validation", 2000, "number of facts to evaluate during validation.")
tf.app.flags.DEFINE_integer("num_consecutive_queries", 1, "number of consecutive queries to supporting evidence.")
#tf.app.flags.DEFINE_boolean("support", False, "Use supporting evidence.")
tf.app.flags.DEFINE_string("device", "/cpu:0", "Use this device.")
tf.app.flags.DEFINE_string("save_dir", "save/" + time.strftime("%d%m%Y_%H%M%S", time.localtime()),
                           "Where to save model and its configuration, always last will be kept.")
tf.app.flags.DEFINE_string("composition", None, "'LSTM', 'GRU', 'RNN', 'BoW', 'BiLSTM', 'BiGRU', 'BiRNN', 'Conv'")
tf.app.flags.DEFINE_string("init_model_path", None, "Path to model to initialize from.")

FLAGS = tf.app.flags.FLAGS

random.seed(FLAGS.random_seed)
tf.set_random_seed(FLAGS.random_seed)

print("Loading KB ...")
kb = KB()
kb.load(FLAGS.kb)

sampler = BatchSampler(kb, FLAGS.batch_size, "test")

train_dir = os.path.join(FLAGS.save_dir)

i = 0

valid_sampler = BatchSampler(kb, FLAGS.batch_size, "valid", FLAGS.subsample_validation)
test_sampler = BatchSampler(kb, FLAGS.batch_size, "test")

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    print("Creating model ...")
    with tf.device(FLAGS.device):
        max_length = kb.max_context_length
        m = QAModel(FLAGS.size, FLAGS.batch_size, len(kb.vocab), max_length,
                    learning_rate=FLAGS.learning_rate, num_consecutive_queries=1)

    print("Created model: " + m.name())

    best_path = []
    checkpoint_path = os.path.join(train_dir, "model.ckpt")

    previous_accs = list()
    epoch = 0

    if os.path.exists(train_dir) and any("ckpt" in x for x in os.listdir(train_dir)):
        newest = max(map(lambda x: os.path.join(train_dir, x),
                         filter(lambda x: not x.endswith(".meta") and "ckpt" in x, os.listdir(train_dir))),
                     key=os.path.getctime)
        print("Loading from checkpoint " + newest)
        best_path.append(newest)
        m.saver.restore(sess, newest)
    else:
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        print("Initializing variables ...")
        sess.run(tf.initialize_all_variables())

    num_params = functools.reduce(lambda acc, x: acc + x.size, sess.run(tf.trainable_variables()), 0)
    print("Num params: %d" % num_params)

    print("Initialized model.")

    def validate():
        # Run evals on development set and print(their perplexity.)
        print("########## Validation ##############")
        acc, mrr = eval.eval_dataset(sess, m, valid_sampler, True)
        print("Accuracy: %.3f" % acc)
        print("MRR: %.3f" % mrr)
        print("####################################")

        if not best_path or acc > max(previous_accs):
            if best_path:
                best_path[0] = m.saver.save(sess, checkpoint_path, global_step=m.global_step, write_meta_graph=False)
            else:
                best_path.append(m.saver.save(sess, checkpoint_path, global_step=m.global_step, write_meta_graph=False))

        if epoch >= 1 and acc <= previous_accs[-1] - 1e-3:  # if mrr is worse by a specific margin
            # if no significant improvement decay learningrate
            print("Decaying learningrate.")
            sess.run(m.learning_rate_decay_op)

        previous_accs.append(acc)

        return acc


    loss = 0.0
    step_time = 0.0
    epoch_acc = 0.0

    while FLAGS.max_iterations < 0 or i < FLAGS.max_iterations:
        i += 1
        start_time = time.time()
        contexts, positions, neg_candidates, supporting_evidence = sampler.get_batch()
        end_of_epoch = sampler.end_of_epoch()
        if end_of_epoch:
            sampler.reset()
        # already fetch next batch parallel to running model
        loss += m.step(sess, contexts, positions, neg_candidates, supporting_evidence, "update")
        step_time += (time.time() - start_time)

        sys.stdout.write("\r%.1f%% Loss: %.3f" %
                         (float((i-1) % FLAGS.ckpt_its + 1.0)*100.0 / FLAGS.ckpt_its,
                          loss / float((i-1) % FLAGS.ckpt_its + 1.0)))
        sys.stdout.flush()

        if end_of_epoch:
            print("")
            epoch += 1
            accuracy = validate()
            print("Epoch %d done!" % epoch)
            if accuracy <= epoch_acc - 1e-3:
                print("Stop learning!")
                break
            else:
                epoch_acc = accuracy

        if i % FLAGS.ckpt_its == 0:
            loss /= FLAGS.ckpt_its
            print("")
            print("%d%% in epoch done." % (100*sampler.get_epoch()))
            # print(statistics for the previous epoch.)
            step_time /= FLAGS.ckpt_its
            print("global step %d learning rate %.5f, step-time %.3f, loss %.4f" % (m.global_step.eval(),
                                                                                    m.learning_rate.eval(),
                                                                                    step_time, loss))
            step_time, loss = 0.0, 0.0
            valid_loss = 0.0

            accuracy = validate()

    best_valid_mrr = max(previous_accs) if previous_accs else 0.0
    print("Restore model to best on validation, with MRR: %.3f" % best_valid_mrr)
    m.saver.restore(sess, best_path[0])
    model_name = best_path[0].split("/")[-1]
    shutil.copyfile(best_path[0], os.path.join(FLAGS.save_dir, model_name))

    print("########## Test ##############")
    acc, mrr = eval.eval_dataset(sess, m, test_sampler, True)
    print("Accuracy: %.3f" % acc)
    print("MRR: %.3f" % mrr)
    print("##############################")
