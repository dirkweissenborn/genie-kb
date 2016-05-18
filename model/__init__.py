from model.models import *
from model.comp_models import *
from model.seq_comp_models import *
from model.comp_functions import *
from data.load_fb15k237 import split_relations


def default_init():
    return tf.random_normal_initializer(0.0, 0.1)


def create_model(kb, size, batch_size, is_train=True, num_neg=200, learning_rate=1e-2,
                 l2_lambda=0.0, is_batch_training=False, model="DistMult",
                 observed_sets=["train_text"], composition=None, num_buckets= 10,
                 comp_util=None, max_vocab_size=10000):
    '''
    Factory Method for all models
    :param model: any or combination of "ModelF", "DistMult", "ModelE", "ModelO", "ModelN"
    :param composition: "Tanh", "LSTM", "GRU", "BiTanh", "BiLSTM", "BiGRU", "BoW" or None
    :return: Model(s) of type "type"
    '''
    if composition and not comp_util:
        comp_util = CompositionUtil(kb, split_relations, max_vocab_size)
    if not isinstance(model, list):
        comp_batch_size = batch_size // (num_neg + 1)
        if not composition:
            composition = ""
        with tf.variable_scope("Comp" + model + "__" + composition):
            comp_size = 2*size if model == "ModelE" else size
            if composition == "RNN":
                composition = TanhRNNCompF(comp_size, comp_batch_size, comp_util, learning_rate)
            elif composition == "LSTM":
                composition = LSTMCompF(comp_size, comp_batch_size, comp_util, learning_rate)
            elif composition == "GRU":
                composition = GRUCompF(comp_size, comp_batch_size, comp_util, learning_rate)
            elif composition == "BiRNN":
                composition = BiTanhRNNCompF(comp_size, comp_batch_size, comp_util, learning_rate)
            elif composition == "BiLSTM":
                composition = BiLSTMCompF(comp_size, comp_batch_size, comp_util, learning_rate)
            elif composition == "BiGRU":
                composition = BiGRUCompF(comp_size, comp_batch_size, comp_util, learning_rate)
            elif composition == "BoW":
                composition = BoWCompF(comp_size, comp_batch_size, comp_util, learning_rate)
            elif composition == "Conv":
                composition = ConvCompF(1, comp_size, comp_batch_size, comp_util, learning_rate)
            else:
                composition = None

        if model == "ModelF":
            return ModelF(kb, size, batch_size, is_train, num_neg, learning_rate, l2_lambda, is_batch_training)
        elif model == "CompBiLSTM":
            return CompBiRNNModel(BasicLSTMCell(size//2), kb, size, batch_size, is_train, num_neg, learning_rate, l2_lambda, is_batch_training)
        elif model == "DistMult":
            if composition:
                return CompDistMult(kb, size, batch_size, composition, is_train, num_neg, learning_rate)
            else:
                return DistMult(kb, size, batch_size, is_train, num_neg, learning_rate, l2_lambda, is_batch_training)
        elif model == "ModelE":
            if composition:
                return CompModelE(kb, size, batch_size, composition, is_train, num_neg, learning_rate)
            else:
                return ModelE(kb, size, batch_size, is_train, num_neg, learning_rate, l2_lambda, is_batch_training)
        elif model == "ModelO":
            if composition:
                return CompModelO(kb, size, batch_size, composition, is_train, num_neg, learning_rate, observed_sets)
            else:
                return ModelO(kb, size, batch_size, is_train, num_neg, learning_rate, l2_lambda, is_batch_training, observed_sets)
        elif model == "WeightedModelO":
            if composition:
                return CompWeightedModelO(kb, size, batch_size, composition, is_train, num_neg, learning_rate, observed_sets)
            else:
                return WeightedModelO(kb, size, batch_size, is_train, num_neg, learning_rate, l2_lambda, is_batch_training, observed_sets)
        elif model == "ModelN":
            return ModelN(kb, size, batch_size, is_train, num_neg, learning_rate, l2_lambda, is_batch_training)
        else:
            raise NameError("There is no model with type %s. "
                            "Possible values are 'ModelF', 'DistMult', 'ModelE', 'ModelO', 'ModelN'." % model)
    else:
        if composition:
            return CompCombinedModel(model, kb, size, batch_size, composition, comp_util, is_train, num_neg,
                                     learning_rate, l2_lambda)
        else:
            return CombinedModel(model, kb, size, batch_size, is_train, num_neg,
                                 learning_rate, l2_lambda, is_batch_training, composition)


def load_model(sess, kb, batch_size, config_file, comp_util=None):
    config = {}
    with open(config_file, 'r') as f:
        for l in f:
            [k, v] = l.strip().split("=")
            config[k] = v
    m = create_model(kb, int(config["size"]), batch_size, is_train=False, composition=config.get("composition"),
                     model=config["model"], comp_util=comp_util)
    m.saver.restore(sess, config["path"])

    return m


def load_models(sess, kb, batch_size, config_files, comp_util=None):
    ms = []
    for config_file in config_files:
        config = {}
        with open(config_file, 'r') as f:
            for l in f:
                [k, v] = l.strip().split("=")
                config[k] = v
        m = create_model(kb, int(config["size"]), batch_size, is_train=False, composition=config.get("composition"),
                         model=config["model"], comp_util=comp_util)
        if not comp_util and hasattr(m, '_comp_f'):
            comp_util = m._comp_f._comp_util
        m.saver.restore(sess, config["path"])
        ms.append(m)

    return ms


#@tf.RegisterGradient("SparseToDense")
#def _tf_sparse_to_dense_grad(op, grad):
#    grad_flat = tf.reshape(grad, [-1])
#    sparse_indices = op.inputs[0]
#    d = tf.gather(tf.shape(sparse_indices), [0])
#    shape = op.inputs[1]
#    cols = tf.gather(shape, [1])
#    ones = tf.expand_dims(tf.ones(d, dtype=tf.int64), 1)
#    cols = ones * cols
#    conc = tf.concat(1, [cols, ones])
#    sparse_indices = tf.reduce_sum(tf.mul(sparse_indices, conc), 1)
#    in_grad = tf.nn.embedding_lookup(grad_flat, sparse_indices)
#    return None, None, in_grad, None