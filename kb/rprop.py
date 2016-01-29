
"""RProp for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer


class RPropOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Adam algorithm.

    @@__init__
    -- (0) get/update state
      local config = config or {}
      local state = state or config
      local stepsize = config.stepsize or 0.1
      local etaplus = config.etaplus or 1.2
      local etaminus = config.etaminus or 0.5
      local stepsizemax = config.stepsizemax or 50.0
      local stepsizemin = config.stepsizemin or 1E-06
      local niter = config.niter or 1
    """

    def __init__(self, stepsize=0.1, etaplus=1.2, etaminus=0.5, stepsizemax=50.0, stepsizemin=1E-06,
                 use_locking=False, name="RProp"):
        super(RPropOptimizer, self).__init__(use_locking, name)
        self._stepsize = stepsize
        self._etaplus = etaplus
        self._etaminus = etaminus
        self._stepsizemax = stepsizemax
        self._stepsizemin = stepsizemin

        # Tensor versions of the constructor arguments, created in _prepare().
        #self._stepsize_t = None
        #self._etaplus_t = None
        #self._etaminus_t = None
        #self._stepsizemax_t = None
        #self._stepsizemin_t = None
        #self._zero_tensor = None



    def _create_slots(self, var_list):
        '''
        state.delta    = dfdx.new(dfdx:size()):zero()
              state.stepsize = dfdx.new(dfdx:size()):fill(stepsize)
              state.sign     = dfdx.new(dfdx:size())
              state.psign    = torch.ByteTensor(dfdx:size())
              state.nsign    = torch.ByteTensor(dfdx:size())
              state.zsign    = torch.ByteTensor(dfdx:size())
              state.dminmax  = torch.ByteTensor(dfdx:size())
              if torch.type(x)=='torch.CudaTensor' then
                  -- Push to GPU
                  state.psign    = state.psign:cuda()
                  state.nsign    = state.nsign:cuda()
                  state.zsign    = state.zsign:cuda()
                  state.dminmax  = state.dminmax:cuda()
              end
        :param var_list:
        :return:
        '''
        # Create the beta1 and beta2 accumulators on the same device as the first
        # variable.

        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "delta", self._name)
            self._get_or_make_slot(v, tf.fill(v.get_shape(), self._stepsize), "stepsize", self._name)
            self._get_or_make_slot(v, tf.zeros([v.get_shape().num_elements()], dtype=tf.float32), "stepmul", self._name)

    def _prepare(self):
        #self._stepsize_t = ops.convert_to_tensor(self._stepsize, name="stepsize")
        #self._etaplus_t = ops.convert_to_tensor(self._etaplus, name="etaplus")
        #self._etaminus_t = ops.convert_to_tensor(self._etaminus, name="etaminus")
        #self._stepsizemax_t = ops.convert_to_tensor(self._stepsizemax, name="stepsizemax")
        #self._stepsizemin_t = ops.convert_to_tensor(self._stepsizemin, name="stepsizemin")
        pass

    def _apply_dense(self, grad, var):
        last_grad = self.get_slot(var, "delta")
        step = self.get_slot(var, "stepsize")
        stepmul = self.get_slot(var, "stepmul")

        sign = tf.sign(last_grad * grad)
        sign_flat = tf.reshape(sign, [-1])

        one_indices = tf.where(tf.equal(sign_flat, 1))
        m_one_indices = tf.where(tf.equal(sign_flat, -1))
        zero_indices = tf.where(tf.equal(sign_flat, 0))

        eta_plus_update = tf.cast(one_indices, tf.float32) * 0.0 + self._etaplus
        eta_minus_update = tf.cast(m_one_indices, tf.float32) * 0.0 + self._etaminus
        zero_update = tf.cast(zero_indices, tf.float32) * 0.0 + 1

        stepmul_up = tf.scatter_update(stepmul, one_indices, eta_plus_update)
        stepmul_up = tf.scatter_update(stepmul_up, m_one_indices, eta_minus_update)
        stepmul_up = tf.scatter_update(stepmul_up, zero_indices, zero_update)

        new_step = step * tf.reshape(stepmul_up, tf.shape(step))
        new_step = tf.maximum(new_step, self._stepsizemin)
        new_step = tf.minimum(new_step, self._stepsizemax)

        step_a = step.assign(new_step)
        up = new_step * tf.sign(grad)
        var_update = var.assign_sub(up, use_locking=self._use_locking)
        with ops.control_dependencies([var_update]):
            self.last_grad_update = last_grad.assign(grad)
        return tf.group(*[var_update, step_a, stepmul_up])


    def _apply_sparse(self, grad, var):
        raise NotImplementedError("RProp should be used only in batch_mode.")
