import tensorflow as tf

def default_init():
    return tf.random_uniform_initializer(0.0, 0.1)
