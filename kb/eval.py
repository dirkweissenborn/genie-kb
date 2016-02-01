import numpy as np
import sys
import math


def rank_triple(sess, kb, model, triple, position="obj"):
    (rel, subj, obj) = triple

    if position == "obj":
        compatible = kb.compatible_args_of(2, rel)
        if obj not in compatible:
            return float('Inf')

        neg_triples = map(lambda e: (rel, subj, e), filter(lambda e:
                                                           e != obj and
                                                           not kb.contains_fact(True, "train", rel, subj, e) and
                                                           not kb.contains_fact(True, "test", rel, subj, e) and
                                                           not kb.contains_fact(True, "valid", rel, subj, e),
                                                           compatible))
    else:
        compatible = kb.compatible_args_of(1, rel)
        if subj not in compatible:
            return float('Inf')
        neg_triples = map(lambda e: (rel, e, obj), filter(lambda e:
                                                          e != subj and
                                                          not kb.contains_fact(True, "train", rel, e, obj) and
                                                          not kb.contains_fact(True, "test", rel, e, obj) and
                                                          not kb.contains_fact(True, "valid", rel, e, obj),
                                                          compatible))

    scores = model.score_triples(sess, [triple] + neg_triples)
    ix = np.argsort(scores)[::-1]
    rank = np.where(ix == 0)[0][0] + 1

    return rank


def eval_triples(sess, kb, model, triples, position="both", verbose=False):
    has_text_mention = set()
    for (pred, subj, obj), _, _ in kb.get_all_facts_of_arity(2, "train_text"):
        has_text_mention.add((subj, obj))
        has_text_mention.add((obj, subj))

    top10 = 0.0
    rec_rank = 0.0
    total = len(triples)
    # with textual mentions
    top10_wt = 0.0
    rec_rank_wt = 0.0
    total_wt = 0.0
    # without textual mentions
    top10_nt = 0.0
    rec_rank_nt = 0.0
    total_nt = 0.0

    ct = 0.0
    i = 0
    for triple in triples:
        i += 1
        if position == "both":
            rank_s = rank_triple(sess, kb, model, triple, "subj")
            rank_o = rank_triple(sess, kb, model, triple, "obj")
            rec_rank += 1.0/rank_s
            rec_rank += 1.0/rank_o
            if rank_s <= 10:
                top10 += 1
            if rank_o <= 10:
                top10 += 1
            ct += 2.0

            if (triple[1], triple[2]) in has_text_mention:
                rec_rank_wt += 1.0/rank_s
                rec_rank_wt += 1.0/rank_o
                if rank_s <= 10:
                    top10_wt += 1
                if rank_o <= 10:
                    top10_wt += 1
                total_wt += 2.0
            else:
                rec_rank_nt += 1.0/rank_s
                rec_rank_nt += 1.0/rank_o
                if rank_s <= 10:
                    top10_nt += 1
                if rank_o <= 10:
                    top10_nt += 1
                total_nt += 2.0
        else:
            rank = rank_triple(sess, kb, model, triple, position)
            rec_rank += 1.0 / rank
            if rank <= 10:
                top10 += 1
            ct += 1.0
            if (triple[1], triple[2]) in has_text_mention:
                rec_rank_wt += 1.0/rank
                if rank <= 10:
                    top10_wt += 1
                total_wt += 1.0
            else:
                rec_rank_nt += 1.0/rank
                if rank <= 10:
                    top10_nt += 1
                total_nt += 1.0
        if verbose:
            if ct % 10 == 0:
                sys.stdout.write("\r%.1f%%, mrr: %.3f, top10: %.3f" % (i*100.0 / total, rec_rank / ct, top10 / ct))
                sys.stdout.flush()

    print ""

    mrr = rec_rank / ct
    top10 /= ct

    if total_wt > 0.0:
        mrr_wt = rec_rank_wt / total_wt
        top10_wt /= total_wt
    else:
        mrr_wt = 0.0

    mrr_nt = rec_rank_nt / total_nt
    top10_nt /= total_nt

    if verbose:
        print "MRR: %.3f" % mrr
        print "Top10: %.3f" % top10
        print "MRR wt: %.3f" % mrr_wt
        print "Top10 wt: %.3f" % top10_wt
        print "MRR nt: %.3f" % mrr_nt
        print "Top10 nt: %.3f" % top10_nt

    return (mrr, top10), (mrr_wt, top10_wt), (mrr_nt, top10_nt)


if __name__ == "__main__":
    import os
    from data.load_fb15k237 import load_fb15k, load_fb15k_type_constraints
    from model.models import *

    # data loading specifics
    tf.app.flags.DEFINE_string('fb15k_dir', None, 'data dir containing files of fb15k dataset')
    # model parameters
    tf.app.flags.DEFINE_integer('size', 10, 'num of models hidden dim')

    # Evaluation
    tf.app.flags.DEFINE_string("model_path", None, "Path to trained model.")
    tf.app.flags.DEFINE_integer("batch_size", 20000, "Number of examples in each batch for training.")
    tf.app.flags.DEFINE_boolean("type_constraint", False, "Use type constraint during sampling.")

    FLAGS = tf.app.flags.FLAGS

    kb = load_fb15k(FLAGS.fb15k_dir,  with_text=False)
    print("Loaded data.")
    if FLAGS.type_constraint:
        print("Loading type constraints!")
        load_fb15k_type_constraints(kb, os.path.join(FLAGS.fb15k_dir, "types"))

    with tf.Session() as sess:
        model = DistMult(kb, FLAGS.size, FLAGS.batch_size, is_train=False)
        model.saver.restore(sess, os.path.join(FLAGS.model_path))
        print("Loaded model.")

        eval_triples(sess, kb, model, map( lambda x: x[0], kb.get_all_facts_of_arity(2, "test")), verbose=True)


