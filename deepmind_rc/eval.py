import numpy as np
import sys

def rank_batch(sess, model, batch):
    contexts, positions, neg_candidates, supporting_evidence = batch
    scores = model.score_examples_with_negs(sess, contexts, positions, neg_candidates, supporting_evidence)
    ix = np.argsort(scores, 1)[:,::-1]
    rank = np.where(ix == 0)[1] + 1

    return rank


def eval_dataset(sess, model, sampler, verbose=False):
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
        print("Accuracy: %.3f" % acc)
        print("MRR: %.3f" % mrr)

    return acc, mrr


if __name__ == "__main__":
    import os
    from data.load_fb15k237 import load_fb15k, load_fb15k_type_constraints
    import prediction_model as model
    import tensorflow as tf
    # data loading specifics
    tf.app.flags.DEFINE_string('fb15k_dir', None, 'data dir containing files of fb15k dataset')
    # model parameters
    tf.app.flags.DEFINE_integer('size', 10, 'num of models hidden dim')
    tf.app.flags.DEFINE_integer('max_vocab', 10000, 'vocab size when using composition.')
    tf.app.flags.DEFINE_string("model", None, "Type of model.")
    tf.app.flags.DEFINE_string("composition", None, "Composition.")
    tf.app.flags.DEFINE_string("observed_sets", "train_text", "Observed sets.")
    tf.app.flags.DEFINE_boolean("support", False, "Use supporting evidence.")
    # Evaluation
    tf.app.flags.DEFINE_string("model_path", None, "Path to trained model.")
    tf.app.flags.DEFINE_string("device", "/cpu:0", "Device to use.")
    tf.app.flags.DEFINE_boolean("type_constraint", False, "Use type constraint during sampling.")
    tf.app.flags.DEFINE_boolean("kb_only", False, "Use only kb relations.")

    FLAGS = tf.app.flags.FLAGS
    FLAGS.observed_sets = FLAGS.observed_sets.split(",")
    print("Loading data...")
    kb = load_fb15k(FLAGS.fb15k_dir, with_text=not FLAGS.kb_only)
    if FLAGS.type_constraint:
        print("Loading type constraints...")
        load_fb15k_type_constraints(kb, os.path.join(FLAGS.fb15k_dir, "types"))

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        with tf.device(FLAGS.device):
            print("Creating model...")
            m = model.create_model(kb, FLAGS.size, 1, is_train=False, learning_rate=0.1,
                                   model=FLAGS.model, observed_sets=FLAGS.observed_sets,
                                   composition=FLAGS.composition,
                                   max_vocab_size=FLAGS.max_vocab,
                                   support=FLAGS.support)

        print("Loading model...")
        m.saver.restore(sess, FLAGS.model_path)
        print("Loaded model.")

        eval_triples(sess, kb, m, [x[0] for x in kb.get_all_facts_of_arity(2, "test")], verbose=True)


