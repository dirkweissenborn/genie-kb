import numpy as np
import sys
import tac_edl as util
import model. query as query


def rank_batch(sess, model, batch):
    scores = model.score_examples(sess, batch)
    ix = np.argsort(scores, 1)[:,::-1]
    rank = np.where(ix == 0)[1] + 1

    return rank

def eval_accuracy(sess, model, sampler, searcher, verbose=False):
    sampler.reset()
    total = 0
    tp = 0
    rr = 0
    while not sampler.end_of_epoch():
        batch = util.get_batch(sess, sampler, searcher, model)
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

def eval_f1(sess, model, sampler, searcher, answer_of_interest, verbose=False):
    sampler.reset()
    total = 0
    tp = 0
    fp = 0
    total_true = 0
    rr = 0
    while not sampler.end_of_epoch():
        batch = util.get_batch(sess, sampler, searcher, model)
        queries = query.flatten_queries(batch)
        selected = [i for i, q in enumerate(queries) if q.answer == answer_of_interest]
        not_selected = [i for i, q in enumerate(queries) if q.answer != answer_of_interest]
        for i, q in enumerate(queries):
            if q.answer != answer_of_interest:
                #switch correct answer to answer of interest
                k = q.neg_candidates.index(answer_of_interest)
                q.neg_candidates[k] = q.answer
                q.answer = answer_of_interest
        ranked = rank_batch(sess, model, batch)
        tp += np.where(ranked[selected] == 1)[0].size
        fp += np.where(ranked[not_selected] == 1)[0].size
        total_true += len(selected)
        total += ranked.size
        rr += np.sum(1.0 / ranked)
        if verbose:
            prec = tp / (tp + fp + 1e-6)
            rec = tp / (total_true + 1e-6)
            f1 = 2 * prec * rec / (prec + rec + 1e-6)
            sys.stdout.write("\r%.1f%%, prec: %.3f, rec: %.3f, f1: %.3f" % (sampler.get_epoch()*100.0,
                                                                            prec, rec, f1))
            sys.stdout.flush()


    prec = tp / (tp + fp + 1e-6)
    rec = tp / (total_true + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    if verbose:
        print("")
    return prec, rec, f1

