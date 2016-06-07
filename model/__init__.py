import tensorflow as tf

def default_init():
    return tf.random_normal_initializer(0.0, 0.1)
