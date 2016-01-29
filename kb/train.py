import random
import os
import time
from data.load_fb15k237 import load_fb15k
from sampler import *
from eval import eval_triples
from model.models import *
import sys


# data loading specifics
tf.app.flags.DEFINE_string('fb15k_dir', None, 'data dir containing files of fb15k dataset')
tf.app.flags.DEFINE_integer('max_vocab', -1, 'max num of symbols in vocab')

# model
tf.app.flags.DEFINE_integer("size", 10, "hidden size of model")

# training
tf.app.flags.DEFINE_float("learning_rate", 1e-2, "Learning rate.")
tf.app.flags.DEFINE_float("l2_lambda", 0, "L2-regularization rate.")
tf.app.flags.DEFINE_float("tau", 0.1, "Text triple weight.")
tf.app.flags.DEFINE_float("learning_rate_decay", 0.5, "Learning rate decay when loss on validation set does not improve.")
tf.app.flags.DEFINE_integer("num_neg", 200, "Number of negative examples for training.")
tf.app.flags.DEFINE_integer("pos_per_batch", 100, "Number of examples in each batch for training.")
tf.app.flags.DEFINE_integer("max_iterations", -1, "Maximum number of batches during training. -1 means until convergence")
tf.app.flags.DEFINE_integer("ckpt_its", -1, "Number of iterations until running checkpoint. Negative means after every epoch.")
tf.app.flags.DEFINE_integer("random_seed", 1234, "Seed for rng.")
tf.app.flags.DEFINE_boolean("kb_only", False, "Only train on kb relations.")
tf.app.flags.DEFINE_boolean("batch_train", False, "Use batch training.")
tf.app.flags.DEFINE_string("save_dir", "save/" + time.strftime("%d%m%Y_%H%M%S", time.localtime()),
                           "Where to save model and its configuration, always last will be kept.")

FLAGS = tf.app.flags.FLAGS

assert (not FLAGS.batch_train or FLAGS.ckpt_its <= -1), "Do not define checkpoint iterations when doing batch training."

if FLAGS.batch_train:
    print("Batch training!")

random.seed(FLAGS.random_seed)

kb = load_fb15k(FLAGS.fb15k_dir, with_text=not FLAGS.kb_only)
num_kb = 0
num_text = 0

for f in kb.get_all_facts():
    if f[2] == "train":
        num_kb += 1
    elif f[2] == "train_text":
        num_text += 1

print("Loaded data. %d kb triples. %d text_triples." % (num_kb, num_text))
batch_size = (FLAGS.num_neg+1) * FLAGS.pos_per_batch * 2  # x2 because subject and object loss training

fact_sampler = BatchNegTypeSampler(kb, FLAGS.pos_per_batch, which_set="train", neg_per_pos=FLAGS.num_neg)
if not FLAGS.kb_only:
    text_sampler = BatchNegTypeSampler(kb, FLAGS.pos_per_batch, which_set="train_text", neg_per_pos=FLAGS.num_neg, type_constrained=False)
print("Created Samplers.")

train_dir = os.path.join(FLAGS.save_dir, "train")
os.makedirs(train_dir)

i = 0

subsample_validation = map(lambda x: x[0], random.sample(kb.get_all_facts_of_arity(2, "valid"), 2000))
# subsample_validation = kb.get_all_facts_of_arity(2, "valid")

if FLAGS.ckpt_its <= 0:
    print "Setting checkpoint iteration to size of whole epoch."
    FLAGS.ckpt_its = fact_sampler.epoch_size

with tf.Session() as sess:
    model = DistMult(kb, FLAGS.size, batch_size, num_neg=FLAGS.num_neg, learning_rate=FLAGS.learning_rate,
                     l2_lambda=FLAGS.l2_lambda, is_batch_training=FLAGS.batch_train)

    sess.run(tf.initialize_all_variables())
    print("Initialized model.")
    loss = 0.0
    step_time = 0.0
    previous_mrrs = list()
    e = 0
    mode = "update"
    if FLAGS.batch_train:
        mode = "accumulate"

    next_batch = fact_sampler.get_batch_async()

    while FLAGS.max_iterations < 0 or i < FLAGS.max_iterations:
        i += 1
        start_time = time.time()

        pos, negs = next_batch.get()
        end_of_epoch = fact_sampler.end_of_epoch()
        # already fetch next batch parallel to running model
        if FLAGS.kb_only:
            next_batch = fact_sampler.get_batch_async()
        else:
            next_batch = text_sampler.get_batch_async()

        loss += model.step(sess, pos, negs, mode)

        if False and not FLAGS.kb_only:
            sess.run(model.training_weight.assign(FLAGS.tau))
            n = (num_text/num_kb)
            for i in xrange(n):
                pos, negs = next_batch.get()
                # already fetch next batch parallel to running model
                if i < n-1:
                    next_batch = text_sampler.get_batch_async()  # next batch should be from text
                else:
                    next_batch = fact_sampler.get_batch_async()  # next batch should be from facts
                l = model.step(sess, pos, negs, mode)
                loss += l
            sess.run(model.training_weight.assign(1.0))

        step_time += (time.time() - start_time)

        sys.stdout.write("\r%.1f%%" % (float((i-1) % FLAGS.ckpt_its + 1.0)*100.0 / FLAGS.ckpt_its))
        sys.stdout.flush()

        if end_of_epoch:
            print ""
            e += 1
            print "Epoch %d done!" % e
            if FLAGS.batch_train:
                model.acc_l2_gradients(sess)
                loss = model.update(sess)
                model.reset_gradients_and_loss(sess)
        #    if FLAGS.l2_lambda > 0:
        #        print "Running L2 Update"
        #        sess.run(model.l2_update)

        if i % FLAGS.ckpt_its == 0:
            if not FLAGS.batch_train:
                print ""
                print "%d%% done in epoch." % ((i*100)/fact_sampler.epoch_size)
            # Print statistics for the previous epoch.
            loss /= FLAGS.ckpt_its
            step_time /= FLAGS.ckpt_its
            print "global step %d learning rate %.4f step-time %.3f loss %.4f" % (model.global_step.eval(),
                                                                                  model.learning_rate.eval(),
                                                                                  step_time, loss)
            step_time, loss = 0.0, 0.0
            valid_loss = 0.0

            # Run evals on development set and print their perplexity.
            print "########## Validation ##############"
            mrr, top10 = eval_triples(sess, kb, model, subsample_validation, verbose=True)

            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_mrrs) > 2 and mrr < min(previous_mrrs[-2:]):
                lr = model.learning_rate.eval()
                sess.run(model.learning_rate.assign(lr * FLAGS.learning_rate_decay))
                print "Decaying learning rate to: %.4f" % model.learning_rate.eval()

            previous_mrrs.append(mrr)
            # Save checkpoint and zero timer and loss.
            checkpoint_path = os.path.join(train_dir, "model.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            print "####################################"
