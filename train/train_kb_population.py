import tensorflow as tf
import argparse
from data.load_fb15k237 import *

# data loading specifics
tf.app.flags.DEFINE_string('fb15k_dir', None, 'data dir containing files of fb15k dataset')
tf.app.flags.DEFINE_integer('max_vocab', -1, 'max num of symbols in vocab')

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")


FLAGS = tf.app.flags.FLAGS


_buckets = [1, 2, 5, 8, 11, 14, 17, 23, 29]

data, vocab, concept_vocab = load_fb15k(FLAGS.fb15k_dir, FLAGS.max_vocab)

#  index tuples by subject and object
subj_facts = [[] for k in concept_vocab.keys()]
obj_facts = [[] for k in concept_vocab.keys()]
subj_obj_facts = dict()

for triple in data["train"]:
    subj = triple[0]
    obj = triple[2]
    subj_facts[subj].append(triple)
    obj_facts[subj].append(triple)
    tup = (subj, obj)
    if tup not in subj_obj_facts:
        subj_obj_facts[tup] = [triple]
    else:
        subj_obj_facts[tup].append(triple)



