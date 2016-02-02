
"""RProp for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer


class RPropOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Adam algorithm."""

    def __init__(self, stepsize=0.1, etaplus=1.2, etaminus=0.5, stepsizemax=50.0, stepsizemin=1E-06,
                 use_locking=False, name="RProp"):
        super(RPropOptimizer, self).__init__(use_locking, name)
        self._stepsize = stepsize
        self._etaplus = etaplus
        self._etaminus = etaminus
        self._stepsizemax = stepsizemax
        self._stepsizemin = stepsizemin

    def _create_slots(self, var_list):
        '''
        :param var_list:
        :return:
        '''
        # Create the beta1 and beta2 accumulators on the same device as the first
        # variable.

        # Create slots for the first and second moments.
        for v in var_list:
            self._get_or_make_slot(v, tf.fill(v.get_shape(), self._stepsize), "stepsize", self._name)
            self._get_or_make_slot(v, tf.zeros([v.get_shape().num_elements()], dtype=tf.float32), "stepmul", self._name)
            self._get_or_make_slot(v, tf.zeros([v.get_shape().num_elements()], dtype=tf.float32), "delta", self._name)

    def _apply_dense(self, grad, var):
        last_grad = self.get_slot(var, "delta")
        step = self.get_slot(var, "stepsize")
        stepmul = self.get_slot(var, "stepmul")

        grad_flat = tf.reshape(grad, [-1])

        sign = tf.sign(last_grad * grad_flat)

        with tf.control_dependencies([sign]):
            last_grad = last_grad.assign(grad_flat)

        one_indices = tf.where(tf.equal(sign, 1))
        m_one_indices = tf.where(tf.equal(sign, -1))
        zero_indices = tf.where(tf.equal(sign, 0))

        eta_plus_update = tf.cast(one_indices, tf.float32) * 0.0 + self._etaplus
        zero_update = tf.cast(m_one_indices, tf.float32) * 0.0
        eta_minus_update = zero_update + self._etaminus
        one_update = tf.cast(zero_indices, tf.float32) * 0.0 + 1

        stepmul_up = tf.scatter_update(stepmul, one_indices, eta_plus_update)
        stepmul_up = tf.scatter_update(stepmul_up, m_one_indices, eta_minus_update)
        stepmul_up = tf.scatter_update(stepmul_up, zero_indices, one_update)

        new_step = step * tf.reshape(stepmul_up, tf.shape(step))
        new_step = tf.maximum(new_step, self._stepsizemin)
        new_step = tf.minimum(new_step, self._stepsizemax)
        step_a = step.assign(new_step)

        last_grad = tf.scatter_update(last_grad, m_one_indices, zero_update)

        up = new_step * tf.reshape(tf.sign(last_grad), tf.shape(step))
        var_update = var.assign_sub(up, use_locking=self._use_locking)

        return tf.group(*[var_update, step_a])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("RProp should be used only in batch_mode.")
