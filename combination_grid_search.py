from eval import *
from model import *

combination_weights = [0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]


def __create_grid(models):
     # first model has fixed weight of one
    return np.zeros([len(combination_weights)] * (len(models)-1))


def grid_rec_rank_triple(sess, kb, models, triple, position="obj", grid=None):
    if grid is None:
        grid = __create_grid(models)
    (rel, subj, obj) = triple

    if position == "obj":
        compatible = kb.compatible_args_of(2, rel)
        if obj not in compatible:
            grid *= 0.0
            return grid

        neg_triples = map(lambda e: (rel, subj, e), filter(lambda e:
                                                           e != obj and
                                                           not kb.contains_fact(True, "train", rel, subj, e) and
                                                           not kb.contains_fact(True, "test", rel, subj, e) and
                                                           not kb.contains_fact(True, "valid", rel, subj, e),
                                                           compatible))
    else:
        compatible = kb.compatible_args_of(1, rel)
        if subj not in compatible:
            grid *= 0.0
            return grid
        neg_triples = map(lambda e: (rel, e, obj), filter(lambda e:
                                                          e != subj and
                                                          not kb.contains_fact(True, "train", rel, e, obj) and
                                                          not kb.contains_fact(True, "test", rel, e, obj) and
                                                          not kb.contains_fact(True, "valid", rel, e, obj),
                                                          compatible))

    scores = [model.score_triples(sess, [triple] + neg_triples) for model in models]

    it = np.nditer(grid, flags=['multi_index'])
    score = np.zeros_like(scores[0], dtype=scores[0].dtype)
    while not it.finished:
        score *= 0
        index = it.multi_index
        score += scores[0]
        for i, ind in enumerate(index):
            w = combination_weights[ind]
            score += w * scores[i+1]
        ix = np.argsort(score)[::-1]
        rank = np.where(ix == 0)[0][0] + 1.0
        grid[index] = 1.0 / rank
        it.iternext()

    return grid


def grid_eval_triples(sess, kb, models, triples, position="both", verbose=False):
    has_text_mention = set()
    for (pred, subj, obj), _, _ in kb.get_all_facts_of_arity(2, "train_text"):
        has_text_mention.add((subj, obj))
        has_text_mention.add((obj, subj))

    top10 = __create_grid(models)
    rec_rank = __create_grid(models)
    total = len(triples)
    # with textual mentions
    top10_wt = __create_grid(models)
    rec_rank_wt = __create_grid(models)
    total_wt = 0.0
    # without textual mentions
    top10_nt = __create_grid(models)
    rec_rank_nt = __create_grid(models)
    total_nt = 0.0

    ct = 0.0
    i = 0

    if position == "both":
        rec_rank_s = __create_grid(models)
        rec_rank_o = __create_grid(models)
        top10_s = __create_grid(models)
        top10_o = __create_grid(models)
    else:
        rec_rank_p = __create_grid(models)
        top10_p = __create_grid(models)

    for triple in triples:
        i += 1
        if position == "both":
            rec_rank_s = grid_rec_rank_triple(sess, kb, models, triple, "subj", rec_rank_s)
            rec_rank_o = grid_rec_rank_triple(sess, kb, models, triple, "obj", rec_rank_o)
            rec_rank += rec_rank_s
            rec_rank += rec_rank_o

            top10_s = np.greater_equal(rec_rank_s, 0.1, top10_s)
            top10_o = np.greater_equal(rec_rank_o, 0.1, top10_o)

            top10 += top10_s
            top10 += top10_o

            ct += 2.0

            if (triple[1], triple[2]) in has_text_mention:
                rec_rank_wt += rec_rank_s
                rec_rank_wt += rec_rank_o
                top10_wt += top10_s
                top10_wt += top10_o
                total_wt += 2.0
            else:
                rec_rank_nt += rec_rank_s
                rec_rank_nt += rec_rank_o
                top10_nt += top10_s
                top10_nt += top10_o
                total_nt += 2.0
        else:
            rec_rank_p = grid_rec_rank_triple(sess, kb, models, triple, position, rec_rank_p)
            rec_rank += rec_rank_p
            top10_p = np.greater_equal(rec_rank_p, 0.1, top10_p)
            top10 += top10_p
            ct += 1.0
            if (triple[1], triple[2]) in has_text_mention:
                rec_rank_wt += rec_rank_p
                top10_wt += top10_p
                total_wt += 1.0
            else:
                rec_rank_nt += rec_rank_p
                top10_nt += top10_p
                total_nt += 1.0
        if verbose:
            if ct % 100 == 0:
                best = np.unravel_index(np.argmax(rec_rank), rec_rank.shape)
                best_mrr = rec_rank[best]
                best_top10 = top10[best]
                sys.stdout.write("\r%.1f%%, mrr: %.3f, top10: %.3f" % (i*100.0 / total, best_mrr / ct, best_top10 / ct))
                sys.stdout.flush()

    best = np.unravel_index(np.argmax(rec_rank), rec_rank.shape)
    best_mrr = rec_rank[best]
    best_top10 = top10[best]
    sys.stdout.write("\r%.1f%%, mrr: %.3f, top10: %.3f" % (i*100.0 / total, best_mrr / ct, best_top10 / ct))
    sys.stdout.flush()
    print ""

    weights = [1.0]
    for i in best:
        weights.append(combination_weights[i])

    return weights


if __name__ == "__main__":
    import os
    from data.load_fb15k237 import load_fb15k, load_fb15k_type_constraints
    from model.models import *

    # data loading specifics
    tf.app.flags.DEFINE_string('fb15k_dir', None, 'data dir containing files of fb15k dataset')

    # Evaluation
    tf.app.flags.DEFINE_string("model_configs", None, "Path to trained model.")
    tf.app.flags.DEFINE_integer("batch_size", 20000, "Number of examples in each batch for training.")
    tf.app.flags.DEFINE_boolean("type_constraint", False, "Use type constraint during sampling.")
    tf.app.flags.DEFINE_boolean("without_text", False, "Also load text triples.")

    FLAGS = tf.app.flags.FLAGS

    print("Loading KB...")
    kb = load_fb15k(FLAGS.fb15k_dir,  with_text=not FLAGS.without_text)
    if FLAGS.type_constraint:
        print("Loading type constraints...")
        load_fb15k_type_constraints(kb, os.path.join(FLAGS.fb15k_dir, "types"))

    with tf.Session() as sess:
        print("Loading models...")
        models = load_models(sess, kb, FLAGS.batch_size, FLAGS.model_configs.split(","))
        print("Loaded models.")

        print("####### Grid Search on Validation ###########")
        weights = grid_eval_triples(sess, kb, models, map(lambda x: x[0], kb.get_all_facts_of_arity(2, "valid")), verbose=True)
        print("Best weights:")
        print(weights)
        print("")
        print("####### Test ###########")
        eval_triples(sess, kb, list(zip(models, weights)), map(lambda x: x[0], kb.get_all_facts_of_arity(2, "test")), verbose=True)