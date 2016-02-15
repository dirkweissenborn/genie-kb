import tensorflow as tf


def batch_dot(t1, t2):
    t1_e = tf.expand_dims(t1, 1)
    t2_e = tf.expand_dims(t2, 2)
    return tf.squeeze(tf.batch_matmul(t1_e, t2_e), [1, 2])


def _clip_by_value(gradients, min_value, max_value):
    # clipping would break IndexedSlices and therefore sparse updates, because they get converted to tensors
    return [tf.clip_by_value(g, min_value, max_value) if isinstance(g, ops.IndexedSlices) else g for g in gradients]