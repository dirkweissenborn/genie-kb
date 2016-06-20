import numpy as np
import sys

def rank_batch(sess, model, batch):
    scores = model.score_examples(sess, batch)
    ix = np.argsort(scores, 1)[:,::-1]
    rank = np.where(ix == 0)[1] + 1
    return rank


def eval_dataset(sess, model, sampler, verbose=False):
    model.set_eval(sess)
    sampler.reset()
    total = 0
    tp = 0
    rr = 0
    while not sampler.end_of_epoch():
        batch = sampler.get_batch()
        ranked = rank_batch(sess, model, batch)
        tp += np.where(ranked == 1)[0].size
        total += ranked.size
        rr += np.sum(1.0 / ranked)
        if verbose:
            sys.stdout.write("\r%.1f%%, acc: %.3f, mrr: %.3f" % (sampler.get_epoch()*100.0, tp / total, rr / total))
            sys.stdout.flush()

    acc = tp / total
    mrr = rr / total
    if verbose:
        print("")
    return acc, mrr

if __name__ == '__main__':
    from kb import KB
    from deepmind_rc.sampler import *
    from model.models import QAModel
    import tensorflow as tf
    import sys
    import functools
    import numpy as np

    # data loading specifics
    tf.app.flags.DEFINE_string('kb', None, 'Path to prepared RC KB.')

    # model
    tf.app.flags.DEFINE_integer("size", 256, "hidden size of model")
    tf.app.flags.DEFINE_integer("max_queries", 2, "max queries to supporting evidence")
    tf.app.flags.DEFINE_integer("num_queries", 1, "num queries to supporting evidence")
    tf.app.flags.DEFINE_integer("max_vocab", -1, "maximum vocabulary size")
    tf.app.flags.DEFINE_integer("batch_size", 25, "Number of examples in each batch for training.")
    tf.app.flags.DEFINE_string("devices", "/cpu:0", "Use this device.")
    tf.app.flags.DEFINE_string("model_file", None, "Model to load.")
    tf.app.flags.DEFINE_string("composition", None, "'LSTM', 'GRU', 'RNN', 'BoW', 'BiLSTM', 'BiGRU', 'BiRNN', 'Conv'")

    FLAGS = tf.app.flags.FLAGS

    print("Loading KB ...")
    kb = KB()
    kb.load(FLAGS.kb)
    valid_sampler = BatchSampler(kb, FLAGS.batch_size, "valid", max_vocab=FLAGS.max_vocab)
    test_sampler = BatchSampler(kb, FLAGS.batch_size, "test", max_vocab=FLAGS.max_vocab)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print("Creating model ...")
        max_length = kb.max_context_length
        devices = FLAGS.devices.split(",")
        vocab_size = min(FLAGS.max_vocab+1, len(kb.vocab)) if FLAGS.max_vocab > 0 else len(kb.vocab)
        m = QAModel(FLAGS.size, FLAGS.batch_size, vocab_size, len(kb.answer_vocab), max_length,
                    max_queries=FLAGS.max_queries, devices=devices)
        print("Created model: " + m.name())
        print("Loading from " + FLAGS.model_file)
        m.saver.restore(sess, FLAGS.model_file)

        print("Consecutive support lookup: %d" % FLAGS.num_queries)
        sess.run(m.num_queries.assign(FLAGS.num_queries))
        num_params = functools.reduce(lambda acc, x: acc + x.size, sess.run(tf.trainable_variables()), 0)
        print("Num params: %d" % num_params)

        print("Initialized model.")
        print("########## Test ##############")
        acc, mrr = eval_dataset(sess, m, test_sampler, True)
        print("Accuracy: %.3f" % acc)
        print("MRR: %.3f" % mrr)
        print("##############################")

        print("########## Valid ##############")
        acc, mrr = eval_dataset(sess, m, valid_sampler, True)
        print("Accuracy: %.3f" % acc)
        print("MRR: %.3f" % mrr)
        print("##############################")
