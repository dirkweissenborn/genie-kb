from data.load_fb15k237 import split_relations
import tensorflow as tf
from prediction_model.models import *
from prediction_model.comp_models import *
from prediction_model.supp_evidence_model import *


def default_init():
    return tf.random_normal_initializer(0.0, 0.1)


def create_model(kb, size, batch_size, is_train=True, learning_rate=1e-2,
                 model="DistMult", observed_sets=["train_text"], composition=None,
                 comp_util=None, max_vocab_size=10000, support=True):
    '''
    Factory Method for all models
    :param model: any or combination of "ModelF", "DistMult", "ModelE", "ModelO", "ModelN"
    :param composition: "Tanh", "LSTM", "GRU", "BiTanh", "BiLSTM", "BiGRU", "BoW" or None
    :return: Model(s) of type "type"
    '''

    if support:
        print("Use supporting facts!")
        m = create_model(kb, size, batch_size, is_train=False, learning_rate=learning_rate,
                         model="DistMult", observed_sets=observed_sets, composition=composition,
                         comp_util=comp_util, max_vocab_size=max_vocab_size, support=False)
        return SupportingEvidenceModel(m, learning_rate, is_train, which_sets=observed_sets)

    if model == "ModelE":
        return ModelE(kb, size, batch_size, is_train, learning_rate)
    elif model == "DistMult":
        if composition is None:
            return DistMult(kb, size, batch_size, is_train, learning_rate)
        else:
            return CompDistMult(kb, size, batch_size, is_train, learning_rate, composition=composition)
    elif model == "CompModel":
        return CompModel(kb, size, batch_size, is_train, learning_rate, composition=composition)
    else:
        raise NameError("There is no model with type %s. "
                        "Possible values are 'ModelF', 'DistMult', 'ModelE', 'ModelO', 'ModelN'." % model)


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
