import os
import time
from kb import KB
from deepmind_rc.sampler import *
from deepmind_rc import eval
from model.models import *
import tensorflow as tf
import sys
import functools
import web.embeddings as embeddings
import numpy as np

# data loading specifics
tf.app.flags.DEFINE_string('kb', None, 'Path to prepared RC KB.')

# model
tf.app.flags.DEFINE_integer("size", 256, "hidden size of model")
tf.app.flags.DEFINE_integer("num_queries", 1, "num queries to supporting evidence")
tf.app.flags.DEFINE_integer("max_vocab", -1, "maximum vocabulary size")
tf.app.flags.DEFINE_float("dropout", 0.0, "Dropout.")
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay", 0.5, "Learning rate decay when loss on validation set does not improve.")
tf.app.flags.DEFINE_integer("batch_size", 25, "Number of examples in each batch for training.")
tf.app.flags.DEFINE_string("devices", "/cpu:0", "Use this device.")
tf.app.flags.DEFINE_string("composition", None, "'LSTM', 'GRU', 'RNN', 'BoW', 'BiLSTM', 'BiGRU', 'BiRNN', 'Conv'")
tf.app.flags.DEFINE_integer("num_models", 1, "Only train single model (1, default) or ensemble with >1 models.")

#training
tf.app.flags.DEFINE_integer("max_iterations", -1, "Maximum number of batches during training. -1 means until convergence")
tf.app.flags.DEFINE_integer("ckpt_its", 1000, "Number of iterations until running checkpoint. Negative means after every epoch.")
tf.app.flags.DEFINE_integer("random_seed", 1234, "Seed for rng.")
tf.app.flags.DEFINE_integer("subsample_validation", -1, "number of facts to evaluate during validation.")
tf.app.flags.DEFINE_string("save_dir", "save/" + time.strftime("%d%m%Y_%H%M%S", time.localtime()),
                           "Where to save model and its configuration, always last will be kept.")
tf.app.flags.DEFINE_string("init_model_path", None, "Path to model to initialize from.")
tf.app.flags.DEFINE_string("embeddings", None, "Init with word embeddings from given path in w2v binary format.")

FLAGS = tf.app.flags.FLAGS

random.seed(FLAGS.random_seed)
tf.set_random_seed(FLAGS.random_seed)

print("Loading KB ...")
kb = KB()
kb.load(FLAGS.kb)

sampler = BatchSampler(kb, FLAGS.batch_size, "train", max_vocab=FLAGS.max_vocab, seed=FLAGS.random_seed+72408)

train_dir = os.path.join(FLAGS.save_dir)
valid_sampler = BatchSampler(kb, FLAGS.batch_size, "valid", FLAGS.subsample_validation, max_vocab=FLAGS.max_vocab)
test_sampler = BatchSampler(kb, FLAGS.batch_size, "test", max_vocab=FLAGS.max_vocab)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    print("Creating model ...")
    max_length = kb.max_context_length
    devices = FLAGS.devices.split(",")
    vocab_size = min(FLAGS.max_vocab+1, len(kb.vocab)) if FLAGS.max_vocab > 0 else len(kb.vocab)
    if FLAGS.num_models > 1:
        qann = EnsembleQAModel(FLAGS.num_models, FLAGS.size, FLAGS.batch_size, vocab_size, vocab_size, max_length,
                               learning_rate=FLAGS.learning_rate, max_queries=FLAGS.num_queries, devices=devices,
                               keep_prob=1.0-FLAGS.dropout)
    else:
        qann = QAModel(FLAGS.size, FLAGS.batch_size, vocab_size, len(kb.answer_vocab), max_length,
                       learning_rate=FLAGS.learning_rate, max_queries=FLAGS.num_queries, devices=devices,
                       keep_prob=1.0-FLAGS.dropout)

    print("Created model: " + qann.name())

    best_path = []
    checkpoint_path = os.path.join(train_dir, "model.ckpt")

    previous_accs = list()
    epoch = 0
    is_ensemble = isinstance(qann, EnsembleQAModel)

    if os.path.exists(train_dir) and any("ckpt" in x for x in os.listdir(train_dir)):
        newest = max(map(lambda x: os.path.join(train_dir, x),
                         filter(lambda x: not x.endswith(".meta") and "ckpt" in x, os.listdir(train_dir))),
                     key=os.path.getctime)
        print("Loading from checkpoint " + newest)
        qann.saver.restore(sess, newest)
    else:
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        print("Initializing variables ...")
        sess.run(tf.initialize_all_variables())
        if FLAGS.embeddings is not None:
            print("Init embeddings with %s..." % FLAGS.embeddings)
            e = embeddings.load_embedding(FLAGS.embeddings)
            if is_ensemble:
                for m in qann.models:
                    em = sess.run(m.input_embedding)
                    for j in range(vocab_size):
                        w = kb.vocab[j]
                        v = e.get(w)
                        if v is not None:
                            em[j, :v.shape[0]] = v
                    sess.run(m.input_embedding.assign(em))
            else:
                em = sess.run(qann.input_embedding)
                for j in range(vocab_size):
                    w = kb.vocab[j]
                    v = e.get(w)
                    if v is not None:
                        em[j, :v.shape[0]] = v
                sess.run(qann.input_embedding.assign(em))

    print("Consecutive support lookup: %d" % FLAGS.num_queries)
    qann.set_num_queries(sess, FLAGS.num_queries)

    num_params = functools.reduce(lambda acc, x: acc + x.size, sess.run(tf.trainable_variables()), 0)
    print("Num params: %d" % num_params)

    print("Initialized model.")

    def validate():
        # Run evals on development set and print(their perplexity.)
        print("########## Validation ##############")
        acc, mrr = eval.eval_dataset(sess, qann, valid_sampler, True)
        print("Accuracy: %.3f" % acc)
        print("MRR: %.3f" % mrr)
        print("####################################")

        if not best_path or acc > max(previous_accs):
            if best_path:
                best_path[0] = qann.saver.save(sess, checkpoint_path, global_step=qann.global_step, write_meta_graph=False)
            else:
                best_path.append(qann.saver.save(sess, checkpoint_path, global_step=qann.global_step, write_meta_graph=False))

        if epoch >= 1 and previous_accs and acc <= previous_accs[-1] - 1e-3:  # if mrr is worse by a specific margin
            # if no significant improvement decay learningrate
            print("Decaying learningrate.")
            qann.set_learning_rate(sess, qann.learning_rate(sess) * FLAGS.learning_rate_decay)

        previous_accs.append(acc)
        qann.set_train(sess)

        return acc


    loss = 0.0
    step_time = 0.0
    epoch_acc = 0.0
    i = 0
    qann.set_train(sess)
    while FLAGS.max_iterations < 0 or i < FLAGS.max_iterations:
        i += 1
        start_time = time.time()
        batch = sampler.get_batch()
        end_of_epoch = sampler.end_of_epoch()
        if end_of_epoch:
            sampler.reset()
        # already fetch next batch parallel to running model
        loss += qann.step(sess, batch, "update")
        step_time += (time.time() - start_time)

        sys.stdout.write("\r%.1f%% Loss: %.3f" %
                         (float((i-1) % FLAGS.ckpt_its + 1.0)*100.0 / FLAGS.ckpt_its,
                          loss / float((i-1) % FLAGS.ckpt_its + 1.0)))
        sys.stdout.flush()

        if end_of_epoch:
            print("")
            epoch += 1
            print("Epoch %d done!" % epoch)

            if not is_ensemble or epoch % qann.num_models == 0:
                accuracy = validate()
                # an ensemble epoch scales with the number of models
                if accuracy <= epoch_acc + 1e-4:
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
            print("global s"
                  "tep %d learning rate %.5f, step-time %.3f, loss %.4f" % (qann.global_step.eval(),
                                                                            qann.learning_rate(sess),
                                                                            step_time, loss))
            step_time, loss = 0.0, 0.0
            valid_loss = 0.0

            accuracy = validate()

    best_valid_acc = max(previous_accs) if previous_accs else 0.0
    print("Restore model to best on validation, with Accuracy: %.3f" % best_valid_acc)
    qann.saver.restore(sess, best_path[0])
    model_name = best_path[0].split("/")[-1]
    print("########## Test ##############")
    acc, mrr = eval.eval_dataset(sess, qann, test_sampler, True)
    print("Accuracy: %.3f" % acc)
    print("MRR: %.3f" % mrr)
    print("##############################")
