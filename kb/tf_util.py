import tensorflow as tf

def dot(t1, t2):
    t1_e = tf.expand_dims(t1, 1)
    t2_e = tf.expand_dims(t2, 2)
    return tf.squeeze(tf.batch_matmul(t1_e, t2_e), [1,2])
