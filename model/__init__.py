from data.load_fb15k237 import split_relations
import tensorflow as tf

def default_init():
    return tf.random_normal_initializer(0.0, 0.1)

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



def create_model(kb, size, batch_size, is_train=True, num_neg=200, learning_rate=1e-2,
                 model="DistMult", observed_sets=["train_text"], composition=None,
                 comp_util=None, max_vocab_size=10000):
    '''
    Factory Method for all models
    :param model: any or combination of "ModelF", "DistMult", "ModelE", "ModelO", "ModelN"
    :param composition: "Tanh", "LSTM", "GRU", "BiTanh", "BiLSTM", "BiGRU", "BoW" or None
    :return: Model(s) of type "type"
    '''

    from model import models
    from model import comp_models
    from model import seq_comp_models
    from model import comp_functions

    if composition and not comp_util:
        comp_util = comp_functions.CompositionUtil(kb, split_relations, max_vocab_size)
    if not isinstance(model, list):
        comp_batch_size = batch_size // (num_neg + 1)
        if not composition:
            composition = ""
        with tf.variable_scope("Comp" + model + "__" + composition):
            comp_size = 2*size if model == "ModelE" else size
            if composition == "RNN":
                composition = comp_functions.TanhRNNCompF(comp_size, comp_batch_size, comp_util, learning_rate)
            elif composition == "LSTM":
                composition = comp_functions.LSTMCompF(comp_size, comp_batch_size, comp_util, learning_rate)
            elif composition == "GRU":
                composition = comp_functions.GRUCompF(comp_size, comp_batch_size, comp_util, learning_rate)
            elif composition == "BiRNN":
                composition = comp_functions.BiTanhRNNCompF(comp_size, comp_batch_size, comp_util, learning_rate)
            elif composition == "BiLSTM":
                composition = comp_functions.BiLSTMCompF(comp_size, comp_batch_size, comp_util, learning_rate)
            elif composition == "BiGRU":
                composition = comp_functions.BiGRUCompF(comp_size, comp_batch_size, comp_util, learning_rate)
            elif composition == "BoW":
                composition = comp_functions.BoWCompF(comp_size, comp_batch_size, comp_util, learning_rate)
            elif composition == "Conv":
                composition = comp_functions.ConvCompF(1, comp_size, comp_batch_size, comp_util, learning_rate)
            else:
                composition = None

        if model == "ModelF":
            return models.ModelF(kb, size, batch_size, is_train, num_neg, learning_rate)
        elif model == "DistMult":
            if composition:
                return comp_models.CompDistMult(kb, size, batch_size, composition, is_train, num_neg, learning_rate)
            else:
                return models.DistMult(kb, size, batch_size, is_train, num_neg, learning_rate)
        elif model == "ModelE":
            if composition:
                return comp_models.CompModelE(kb, size, batch_size, composition, is_train, num_neg, learning_rate)
            else:
                return models.ModelE(kb, size, batch_size, is_train, num_neg, learning_rate)
        elif model == "ModelO":
            if composition:
                return comp_models.CompModelO(kb, size, batch_size, composition, is_train, num_neg, learning_rate, observed_sets)
            else:
                return models.ModelO(kb, size, batch_size, is_train, num_neg, learning_rate, observed_sets)
        elif model == "WeightedModelO":
            if composition:
                return comp_models.CompWeightedModelO(kb, size, batch_size, composition, is_train, num_neg, learning_rate, observed_sets)
            else:
                return models.WeightedModelO(kb, size, batch_size, is_train, num_neg, learning_rate, observed_sets)
        elif model == "ModelN":
            return models.ModelN(kb, size, batch_size, is_train, num_neg, learning_rate)
        else:
            raise NameError("There is no model with type %s. "
                            "Possible values are 'ModelF', 'DistMult', 'ModelE', 'ModelO', 'ModelN'." % model)
    else:
        if composition:
            return comp_models.CompCombinedModel(model, kb, size, batch_size, composition, comp_util, is_train, num_neg,
                                                 learning_rate)
        else:
            return models.CombinedModel(model, kb, size, batch_size, is_train, num_neg,
                                        learning_rate, composition)