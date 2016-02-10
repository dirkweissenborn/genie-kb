import tensorflow as tf
from models import *
from comp_models import *
from data.load_fb15k237 import split_relations

def default_init():
    return tf.random_normal_initializer(0.0, 0.1)


def create_model(kb, size, batch_size, is_train=True, num_neg=200, learning_rate=1e-2,
                 l2_lambda=0.0, is_batch_training=False, type="DistMult",
                 observed_sets=["train_text"], composition=None, num_buckets= 10):
    '''
    Factory Method for all models
    :param type: any or combination of "ModelF", "DistMult", "ModelE", "ModelO", "ModelN"
    :param composition: "Tanh", "LSTM", "GRU", "BiTanh", "BiLSTM", "BiGRU", "BoW" or None
    :return: Model(s) of type "type"
    '''
    if not isinstance(type, list):
        with vs.variable_scope(type):
            comp_size = 2*size if type == "ModelE" else size
            if composition == "Tanh":
                composition = TanhRNNCompModel(kb, comp_size, num_buckets, split_relations, batch_size/(num_neg+1), learning_rate)
            elif composition == "LSTM":
                composition = LSTMCompModel(kb, comp_size, num_buckets, split_relations, batch_size/(num_neg+1), learning_rate)
            elif composition == "GRU":
                composition = GRUCompModel(kb, comp_size, num_buckets, split_relations, batch_size/(num_neg+1), learning_rate)
            elif composition == "BiTanh":
                composition = BiTanhRNNCompModel(kb, comp_size, num_buckets, split_relations, batch_size/(num_neg+1), learning_rate)
            elif composition == "BiLSTM":
                composition = BiLSTMCompModel(kb, comp_size, num_buckets, split_relations, batch_size/(num_neg+1), learning_rate)
            elif composition == "BiGRU":
                composition = BiGRUCompModel(kb, comp_size, num_buckets, split_relations, batch_size/(num_neg+1), learning_rate)
            elif composition == "BoW":
                composition = CompositionModel(kb, comp_size, num_buckets, split_relations, batch_size/(num_neg+1), learning_rate)
            else:
                composition = None
            if type == "ModelF":
                return ModelF(kb, size, batch_size, is_train, num_neg, learning_rate, l2_lambda, is_batch_training)
            elif type == "DistMult":
                if composition:
                    return CompDistMult(kb, size, batch_size, composition, is_train, num_neg, learning_rate)
                else:
                    return DistMult(kb, size, batch_size, is_train, num_neg, learning_rate, l2_lambda, is_batch_training)
            elif type == "ModelE":
                if composition:
                    return CompModelE(kb, size, batch_size, composition, is_train, num_neg, learning_rate)
                else:
                    return ModelE(kb, size, batch_size, is_train, num_neg, learning_rate, l2_lambda, is_batch_training)
            elif type == "ModelO":
                return ModelO(kb, size, batch_size, is_train, num_neg, learning_rate, l2_lambda, is_batch_training, observed_sets)
            elif type == "ModelN":
                return ModelN(kb, size, batch_size, is_train, num_neg, learning_rate, l2_lambda, is_batch_training, observed_sets)
            else:
                raise NameError("There is no model with type %s. "
                                "Possible values are 'ModelF', 'DistMult', 'ModelE', 'ModelO', 'ModelN'." % type)
    else:
        if composition:
            return CompCombinedModel(type, kb, size, batch_size, is_train, num_neg,
                                     learning_rate, l2_lambda, is_batch_training, composition)
        else:
            return CombinedModel(type, kb, size, batch_size, is_train, num_neg,
                                 learning_rate, l2_lambda, is_batch_training, composition)