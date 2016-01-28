import numpy as np
import sys

def rank_triple(sess, kb, model, triple, position="obj"):
    dim = 2 if position == "obj" else 1
    (rel, subj, obj) = triple

    neg_triples = filter(lambda e:
                         not kb.contains_fact(True, "train", rel, subj, e) if position == "obj"
                         else not kb.contains_fact(True, "train", rel, e, obj),
                         kb.get_symbols(dim))

    neg_triples = map(lambda e: (rel, subj, e) if position == "obj" else (rel, e, obj), neg_triples)
    scores = np.array(model.score_triples(sess, [triple] + neg_triples))
    ix = np.argsort(scores)[::-1]
    rank = np.where(ix == 0)[0][0] + 1

    return rank, ix


def eval_triples(sess, kb, model, triples, position="obj", verbose=False):
    total = len(triples)
    top10 = 0.0
    rec_rank = 0.0

    ct = 0.0

    for triple in triples:
        rank, _ = rank_triple(sess, kb, model, triple, position)
        rec_rank += 1.0 / rank
        if rank <= 10:
            top10 += 1
        if verbose:
            ct += 1.0
            if ct % 10 == 0:
                sys.stdout.write("\r%.1f%%, mrr: %.3f, top10: %.3f" % (ct*100.0 / total, rec_rank / ct, top10 / ct))
                sys.stdout.flush()

    print ""

    mrr = rec_rank / total
    top10 /= total

    if verbose:
        print "MRR: %.3f" % mrr
        print "Top10: %.3f" % top10

    return mrr, top10


if __name__ == "__main__":
    import os
    from data.load_fb15k237 import load_fb15k
    from model.models import *

    # data loading specifics
    tf.app.flags.DEFINE_string('fb15k_dir', None, 'data dir containing files of fb15k dataset')

    # Evaluation
    tf.app.flags.DEFINE_string("model_path", None, "Path to trained model.")
    tf.app.flags.DEFINE_integer("batch_size", 20000, "Number of examples in each batch for training.")

    FLAGS = tf.app.flags.FLAGS

    kb = load_fb15k(FLAGS.fb15k_dir,  with_text=False)
    print("Loaded data.")

    with tf.Session() as sess:
        model = DistMult(kb, 10, FLAGS.batch_size, is_train=False)
        model.saver.restore(sess, os.path.join(FLAGS.model_path))
        print("Loaded model.")

        eval_triples(sess, kb, model, map( lambda x: x[0], kb.get_all_facts_of_arity(2, "test")), verbose=True)


