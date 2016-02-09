from model.models import *
import tensorflow as tf
class CompositionalKBScoringModel(AbstractKBScoringModel):

    def __init__(self, kb, size, batch_size, relation2sequence, comp_f, is_train=True, num_neg=200, learning_rate=1e-2, l2_lambda=0.0,
                 is_batch_training=False):
        AbstractKBScoringModel.__init__(self, kb, size, batch_size, is_train, num_neg,
                                        learning_rate, l2_lambda, is_batch_training)
        self._rel2seq = relation2sequence
        self._comp_f = comp_f

    def _init_inputs(self):
        max_l = 0
        num_buckets = 5
        l_count = dict()
        total = 0
        for (rel, _, _), _, typ in self._kb.get_all_facts():
            if typ == "train_text":
                s = self._rel2seq(rel)
                l = len(s)
                max_l = max(max_l, l)
                if l not in l_count:
                    l_count[l] = 0
                l_count[l] += 1
                total += 1

        self._seq_inputs = [tf.placeholder(tf.int64, shape=[None], name="seq_input%d" % i)
                            for i in xrange(max_l)]
        with vs.variable_scope("composition", initializer=self._init):
            seq_outputs = self._comp_f(self._seq_inputs)
        self._seq_outputs = []
        ct = 0
        for l in xrange(max(l_count.keys())):
            c = l_count.get(l)
            if c:
                ct += c
                if ct % (total / num_buckets) < c:
                    self._seq_outputs.append(seq_outputs[:l])
        train_vars = tf.trainable_variables()
        self._seq_grad = tf.placeholder(tf.int64, shape=[None, self._size], name="rel_grad")
        self._rel_input = tf.placeholder(tf.int64, shape=[None, self._size], name="rel")
        self._seq_bp = [self.opt.apply_gradients(zip(tf.gradients(os[-1], train_vars, self._seq_grad),
                                                     train_vars))
                        for os in self._seq_outputs]


