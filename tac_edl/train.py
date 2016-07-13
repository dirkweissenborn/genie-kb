import os
import time
from kb import FactKB
from tac_edl.candidate_search import ANNSearcher
from tac_edl.sampler import *
import tac_edl as util
from tac_edl import eval
from model.models import QAModel
import tensorflow as tf
import sys
import functools

# data loading specifics
tf.app.flags.DEFINE_string('kb', None, 'Path to prepared TAC EDL KB.')

# model
tf.app.flags.DEFINE_integer("size", 10, "hidden size of model")
tf.app.flags.DEFINE_integer("max_queries", 1, "max queries to supporting evidence")
tf.app.flags.DEFINE_integer("num_queries", 0, "num queries to supporting evidence")

# training
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay", 0.5, "Learning rate decay when loss on validation set does not improve.")
tf.app.flags.DEFINE_integer("batch_size", 1, "Number of examples in each batch for training.")
tf.app.flags.DEFINE_integer("max_iterations", -1, "Maximum number of batches during training. -1 means until convergence")
tf.app.flags.DEFINE_integer("ckpt_its", -1, "Number of iterations until running checkpoint. Negative means after every epoch.")
tf.app.flags.DEFINE_integer("random_seed", 1234, "Seed for rng.")
tf.app.flags.DEFINE_string("devices", "/cpu:0", "Use this device.")
tf.app.flags.DEFINE_string("save_dir", "save/" + time.strftime("%d%m%Y_%H%M%S", time.localtime()),
                           "Where to save model and its configuration, always last will be kept.")
tf.app.flags.DEFINE_string("composition", None, "'LSTM', 'GRU', 'RNN', 'BoW', 'BiLSTM', 'BiGRU', 'BiRNN', 'Conv'")
tf.app.flags.DEFINE_string("init_model_path", None, "Path to model to initialize from.")
tf.app.flags.DEFINE_string("train_sets", "TAC_ED/2014_train,TAC_ET/2014_train", "Path to model to initialize from.")
tf.app.flags.DEFINE_string("valid_sets", "TAC_ED/2014_eval,TAC_ET/2014_eval", "Path to model to initialize from.")
#tf.app.flags.DEFINE_string("test_sets", "TAC_KB,EL/2014_train,", "Path to model to initialize from.")

FLAGS = tf.app.flags.FLAGS

random.seed(FLAGS.random_seed)
tf.set_random_seed(FLAGS.random_seed)

print("Loading KB ...")
fact_kb = FactKB()
fact_kb.load(FLAGS.kb)

train_sets = FLAGS.train_sets.split(",")
valid_sets = FLAGS.valid_sets.split(",")

train_dir = FLAGS.save_dir
i = 0

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    print("Creating model ...")
    max_length = fact_kb.kb.max_context_length
    devices = FLAGS.devices.split(",")
    m = QAModel(FLAGS.size, FLAGS.batch_size, len(fact_kb.kb.vocab), len(fact_kb.entity_vocab), max_length,
                learning_rate=FLAGS.learning_rate, max_queries=FLAGS.max_queries,
                devices=devices)

    print("Created model: " + m.name())

    best_path = []
    checkpoint_path = os.path.join(train_dir, "model.ckpt")

    results = {ds:[0.0] for ds in valid_sets}
    epoch = 0

    if os.path.exists(train_dir) and any("ckpt" in x for x in os.listdir(train_dir)):
        newest = max(map(lambda x: os.path.join(train_dir, x),
                         filter(lambda x: not x.endswith(".meta") and "ckpt" in x, os.listdir(train_dir))),
                     key=os.path.getctime)
        print("Loading from checkpoint " + newest)
        m.saver.restore(sess, newest)
    else:
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        print("Initializing variables ...")
        sess.run(tf.initialize_all_variables())

    print("Consecutive support lookup: %d" % FLAGS.num_queries)
    sess.run(m._num_queries.assign(FLAGS.num_queries))

    num_params = functools.reduce(lambda acc, x: acc + x.size, sess.run(tf.trainable_variables()), 0)
    print("Num params: %d" % num_params)

    searcher = None
    if any(ds.startswith("TAC_EL/") for ds in train_sets):
        searcher = ANNSearcher(sess.run(m.output_embedding))

    sampler = BatchSampler(fact_kb, FLAGS.batch_size, train_sets)
    valid_samplers = {ds:BatchSampler(fact_kb, FLAGS.batch_size, [ds]) for ds in valid_sets}
    #test_sampler = BatchSampler(fact_kb, FLAGS.batch_size, "test")

    print("Initialized model.")

    def validate():
        # Run evals on development set and print(their perplexity.)
        print("########## Validation ##############")
        for ds in valid_sets:
            print(ds)
            if ds.startswith("TAC_ED/"):
                prec, rec, f1 = eval.eval_f1(sess, m, valid_samplers[ds], searcher,
                                             answer_of_interest=fact_kb.id("[ENTITY]"), verbose=True)
                print("Prec: %.3f" % prec)
                print("Rec: %.3f" % rec)
                print("F1: %.3f" % f1)
                results[ds].append(f1)
            else:
                acc, mrr = eval.eval_accuracy(sess, m, valid_samplers[ds], searcher, True)
                print("Accuracy: %.3f" % acc)
                print("MRR: %.3f" % mrr)
                results[ds].append(acc)
            print("####################################")
        if any(ds.startswith("TAC_EL/") for ds in train_sets):
            print("Updating ANN candidate search...")
            searcher.update(sess.run(m.output_embedding))
            print("Done.")

        if not best_path or any(results[ds][-1] > max(results[ds][:-1]) for ds in valid_sets):
            if best_path:
                best_path[0] = m.saver.save(sess, checkpoint_path, global_step=m.global_step, write_meta_graph=False)
            else:
                best_path.append(m.saver.save(sess, checkpoint_path, global_step=m.global_step, write_meta_graph=False))

        if epoch >= 1 and all(results[ds][-1] <= results[ds][-2] - 1e-3 for ds in valid_sets):
            # if results get worse by a specific margin  decay learningrate
            print("Decaying learningrate.")
            sess.run(m._learning_rate.assign(m._learning_rate * FLAGS.learning_rate_decay))


    loss = 0.0
    step_time = 0.0
    epoch_result = {ds:0.0 for ds in valid_sets}

    print("Epoch size: %d" % sampler.epoch_size)

    if FLAGS.ckpt_its <= 0:
        FLAGS.ckpt_its = sampler.epoch_size

    while FLAGS.max_iterations < 0 or i < FLAGS.max_iterations:
        i += 1
        start_time = time.time()
        batch = util.get_batch(sess, sampler, searcher, m)
        end_of_epoch = sampler.end_of_epoch()
        if end_of_epoch:
            sampler.reset()
        # already fetch next batch parallel to running model
        loss += m.step(sess, batch, "update")
        step_time += (time.time() - start_time)

        sys.stdout.write("\r%.1f%% Loss: %.3f" %
                         (float((i-1) % FLAGS.ckpt_its + 1.0)*100.0 / FLAGS.ckpt_its,
                          loss / float((i-1) % FLAGS.ckpt_its + 1.0)))
        sys.stdout.flush()

        if i % FLAGS.ckpt_its == 0:
            loss /= FLAGS.ckpt_its
            print("")
            print("%d%% in epoch done." % (100*sampler.get_epoch()))
            # print(statistics for the previous epoch.)
            step_time /= FLAGS.ckpt_its
            print("global step %d learning rate %.5f, step-time %.3f, loss %.4f" % (m.global_step.eval(),
                                                                                    m._learning_rate.eval(),
                                                                                    step_time, loss))
            step_time, loss = 0.0, 0.0
            valid_loss = 0.0
            if not end_of_epoch:
                validate()

        if end_of_epoch:
            print("")
            epoch += 1
            validate()
            print("Epoch %d done!" % epoch)
            if all(results[ds][-1] <= epoch_result[ds] - 1e-3 for ds in valid_sets):
                print("Stop learning!")
                break
            else:
                for ds in valid_sets:
                    epoch_result[ds] = results[ds][-1]

    print("Restore model to best on validation: %s" % best_path[0])
    m.saver.restore(sess, best_path[0])
    model_name = best_path[0].split("/")[-1]

    print("########## Test ##############")
    #acc, mrr = eval.eval_dataset(sess, m, test_sampler, True)
    acc = mrr = 0.0
    print("Accuracy: %.3f" % acc)
    print("MRR: %.3f" % mrr)
    print("##############################")
