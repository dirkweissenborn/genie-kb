import random
import os
import time
from data.load_fb15k237 import load_fb15k, load_fb15k_type_constraints, split_relations
from sampler import *
from eval import eval_triples
import prediction_model as model
import tensorflow as tf
import sys
from kb import subsample_kb
import shutil
import json
import functools


# data loading specifics
tf.app.flags.DEFINE_string('fb15k_dir', None, 'data dir containing extracted files of fb15k dataset.')
tf.app.flags.DEFINE_integer('max_vocab', 10000, 'max vocabulary when composition is used.')

# model
tf.app.flags.DEFINE_integer("size", 10, "hidden size of model")

# training
tf.app.flags.DEFINE_float("learning_rate", 1e-2, "Learning rate.")
tf.app.flags.DEFINE_float("sample_text_prob", 0.935,
                          "Probability of sampling text triple (default is ratio of text (emnlp) to kb triples.")
tf.app.flags.DEFINE_float("learning_rate_decay", 0.5, "Learning rate decay when loss on validation set does not improve.")
tf.app.flags.DEFINE_integer("num_neg", 200, "Number of negative examples for training.")
tf.app.flags.DEFINE_integer("pos_per_batch", 100, "Number of examples in each batch for training.")
tf.app.flags.DEFINE_integer("max_iterations", -1, "Maximum number of batches during training. -1 means until convergence")
tf.app.flags.DEFINE_integer("ckpt_its", -1, "Number of iterations until running checkpoint. Negative means after every epoch.")
tf.app.flags.DEFINE_integer("random_seed", 1234, "Seed for rng.")
tf.app.flags.DEFINE_integer("subsample_kb", -1, "num of entities in subsampled kb. if <= 0 use whole kb")
tf.app.flags.DEFINE_integer("subsample_validation", 2000, "number of facts to evaluate during validation.")
tf.app.flags.DEFINE_boolean("kb_only", False, "Only load and train on FB relations, ignoring text.")
tf.app.flags.DEFINE_boolean("type_constraint", False, "Use type constraint during sampling.")
tf.app.flags.DEFINE_string("device", "/cpu:0", "Use this device.")
tf.app.flags.DEFINE_string("save_dir", "save/" + time.strftime("%d%m%Y_%H%M%S", time.localtime()),
                           "Where to save model and its configuration, always last will be kept.")
tf.app.flags.DEFINE_string("model", "DistMult",
                           "Model architecture or combination thereof split by comma of: "
                           "'ModelF', 'DistMult', 'ModelE', 'ModelO', 'ModelN', 'WeightedModelO'")
tf.app.flags.DEFINE_string("observed_sets", "train_text", "Which sets to observe for observed models.")
tf.app.flags.DEFINE_string("valid_mode", "a", "[a,t,nt] are possible. a- validate on all triples, "
                                              "t- validate only on triples with text mentions, "
                                              "nt- validate only on triples without text mentions")
tf.app.flags.DEFINE_string("composition", None, "'LSTM', 'GRU', 'RNN', 'BoW', 'BiLSTM', 'BiGRU', 'BiRNN', 'Conv'")

FLAGS = tf.app.flags.FLAGS

if "," in FLAGS.model:
    FLAGS.model = FLAGS.model.split(",")

FLAGS.observed_sets = FLAGS.observed_sets.split(",")

random.seed(FLAGS.random_seed)
tf.set_random_seed(FLAGS.random_seed)

print("Loading KB ...")
kb = load_fb15k(FLAGS.fb15k_dir, with_text=not FLAGS.kb_only)
if FLAGS.subsample_kb > 0:
    kb = subsample_kb(kb, FLAGS.subsample_kb)

if FLAGS.type_constraint:
    print("Loading type constraints...")
    load_fb15k_type_constraints(kb, os.path.join(FLAGS.fb15k_dir, "types"))

num_kb = 0
num_text = 0

for f in kb.get_all_facts():
    if f[2] == "train":
        num_kb += 1
    elif f[2] == "train_text":
        num_text += 1

print("Loaded KB. %d kb triples. %d text_triples." % (num_kb, num_text))
batch_size = FLAGS.pos_per_batch  # x2 because subject and object loss training

fact_sampler = BatchNegTypeSampler(kb, FLAGS.pos_per_batch, which_set="train", neg_per_pos=FLAGS.num_neg, type_constraint=FLAGS.type_constraint)
if not FLAGS.kb_only:
    text_sampler = BatchNegTypeSampler(kb, FLAGS.pos_per_batch, which_set="train_text", neg_per_pos=FLAGS.num_neg, type_constraint=False)
print("Created Samplers.")

train_dir = os.path.join(FLAGS.save_dir, "train")

i = 0

validation = [x[0] for x in kb.get_all_facts_of_arity(2, "valid")]
if len(validation) > FLAGS.subsample_validation > -1:
    validation = random.sample(validation, FLAGS.subsample_validation)


if FLAGS.ckpt_its <= 0:
    print("Setting checkpoint iteration to size of whole epoch.")
    FLAGS.ckpt_its = fact_sampler.epoch_size

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    print("Creating model ...")
    with tf.device(FLAGS.device):
        m = model.create_model(kb, FLAGS.size, batch_size, learning_rate=FLAGS.learning_rate,
                               model=FLAGS.model, observed_sets=FLAGS.observed_sets, composition=FLAGS.composition,
                               max_vocab_size=FLAGS.max_vocab)

    print("Created model: " + m.name())

    if os.path.exists(train_dir) and any("ckpt" in x for x in os.listdir(train_dir)):
        newest = max(map(lambda x: os.path.join(train_dir, x),
                         filter(lambda x: ".ckpt" in x, os.listdir(train_dir))), key=os.path.getctime)
        print("Loading from checkpoint " + newest)
        m.saver.restore(sess, newest)
    else:
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        sess.run(tf.initialize_all_variables())

    num_params = functools.reduce(lambda acc, x: acc + x.size, sess.run(tf.trainable_variables()), 0)
    print("Num params: %d" % num_params)

    print("Initialized model.")

    best_path = []
    checkpoint_path = os.path.join(train_dir, "model.ckpt")

    def validate():
        # Run evals on development set and print(their perplexity.)
        print("########## Validation ##############")
        (mrr_a, _), (mrr_t, _), (mrr_nt, _) = eval_triples(sess, kb, m, validation, verbose=True)

        if FLAGS.valid_mode == "a":
            mrr = mrr_a
        elif FLAGS.valid_mode == "t":
            mrr = mrr_t
        elif FLAGS.valid_mode == "nt":
            mrr = mrr_nt
        else:
            raise ValueError("valid_mode flag must be either 'a','t' or 'nt'")
        print("####################################")

        if not best_path or mrr > max(previous_mrrs):
            if best_path:
                os.remove(best_path[0])
            best_path[0] = m.saver.save(sess, checkpoint_path, global_step=m.global_step)
        previous_mrrs.append(mrr)

        return mrr


    end_of_epoch = False
    def sample_next_batch():
        if FLAGS.kb_only or random.random() >= FLAGS.sample_text_prob:
            return fact_sampler.get_batch_async("obj" if random.random() > 0.5 else "subj")
        else:
            return text_sampler.get_batch_async("obj" if random.random() > 0.5 else "subj")

    next_batch = sample_next_batch()
    loss = 0.0
    step_time = 0.0
    previous_mrrs = list()
    epoch_mrr = 0.0
    e = 0

    while FLAGS.max_iterations < 0 or i < FLAGS.max_iterations:
        i += 1
        start_time = time.time()
        pos, negs, is_inv = next_batch.get()
        end_of_epoch = fact_sampler.end_of_epoch()
        if end_of_epoch:
            fact_sampler.reset()
        if text_sampler.end_of_epoch():
            text_sampler.reset()
        # already fetch next batch parallel to running model
        next_batch = sample_next_batch()

        loss += m.step(sess, pos, negs, is_inv, "update")
        step_time += (time.time() - start_time)

        sys.stdout.write("\r%.1f%% Loss: %.3f" %
                         (float((i-1) % FLAGS.ckpt_its + 1.0)*100.0 / FLAGS.ckpt_its,
                          loss / float((i-1) % FLAGS.ckpt_its + 1.0)))
        sys.stdout.flush()

        if end_of_epoch:
            print("")
            e += 1
            mrr = validate()
            print("Epoch %d done!" % e)
            if mrr <= epoch_mrr - 1e-3:
                print("Stop learning!")
                break
            else:
                epoch_mrr = mrr

        if i % FLAGS.ckpt_its == 0:
            loss /= FLAGS.ckpt_its
            print("")
            print("%d%% in epoch done." % (100*fact_sampler.get_epoch()))
            # print(statistics for the previous epoch.)
            step_time /= FLAGS.ckpt_its
            print("global step %d learning rate %.4f, step-time %.3f, loss %.4f" % (m.global_step.eval(),
                                                                                    m.learning_rate.eval(),
                                                                                    step_time, loss))
            step_time, loss = 0.0, 0.0
            valid_loss = 0.0

            mrr = validate()

            if e >= 1 and mrr <= previous_mrrs[-1] - 1e-3:  # if mrr is worse by a specific margin
                # if no significant improvement decay learningrate
                print("Decaying learningrate.")
                sess.run(m.learning_rate_decay_op)


    best_valid_mrr = max(previous_mrrs)
    print("Restore model to best on validation, with MRR: %.3f" % best_valid_mrr)
    m.saver.restore(sess, best_path)
    model_name = best_path.split("/")[-1]
    shutil.copyfile(best_path[0], os.path.join(FLAGS.save_dir, model_name))
    print("########## Test ##############")
    (mrr, top10), (mrr_wt, top10_wt), (mrr_nt, top10_nt) = \
        eval_triples(sess, kb, m, [x[0] for x in kb.get_all_facts_of_arity(2, "test")], verbose=True)
    with open(os.path.join(FLAGS.save_dir, "result.txt"), 'w') as f:
        f.write("best model: %s\n\nMRR: %.3f\nHits10: %.3f\n\n" % (model_name, mrr, top10))
        f.write("MRR wt: %.3f\nHits10 wt: %.3f\n\n" % (mrr_wt, top10_wt))
        f.write("MRR nt: %.3f\nHits10 nt: %.3f\n\nFLAGS:\n" % (mrr_nt, top10_nt))
        f.write(json.dumps(FLAGS.__flags, sort_keys=True, indent=2, separators=(',', ': ')))
        f.flush()
    with open(os.path.join(FLAGS.save_dir, "model.cfg"), 'w') as f:
        f.write("size=%d\n" % FLAGS.size)
        f.write("path=%s\n" % os.path.join(FLAGS.save_dir, model_name))
        if FLAGS.composition:
            f.write("composition=%s\n" % FLAGS.composition)
        f.write("model=%s" % FLAGS.model)
        f.flush()
    print("##############################")
