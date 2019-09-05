# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
import abc

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K

IMAGE_DATA_FORMAT = K.image_data_format()


def compute_kwinners(x, k, duty_cycles, boost_strength):
    r"""
    Use the boost strength to compute a boost factor for each unit represented
    in x. These factors are used to increase the impact of each unit to improve
    their chances of being chosen. This encourages participation of more columns
    in the learning process.

    The boosting function is a curve defined as: boost_factors = exp[ -
    boost_strength * (dutyCycle - target_density)] Intuitively this means that
    units that have been active (i.e. in the top-k) at the target activation
    level have a boost factor of 1, meaning their activity is not boosted.
    Columns whose duty cycle drops too much below that of their neighbors are
    boosted depending on how infrequently they have been active. Unit that has
    been active more than the target activation level have a boost factor below
    1, meaning their activity is suppressed and they are less likely to be in
    the top-k.

    Note that we do not transmit the boosted values. We only use boosting to
    determine the winning units.

    The target activation density for each unit is k / number of units. The
    boostFactor depends on the dutyCycle via an exponential function::

          boostFactor
              ^
              |
              |\
              | \
        1  _  |  \
              |    _
              |      _ _
              |          _ _ _ _
              +--------------------> dutyCycle
                 |
            target_density

    :param x:
        Current activity of each unit.

    :param k:
        The activity of the top k units will be allowed to remain, the rest are
        set to zero.

    :param duty_cycles:
        The averaged duty cycle of each unit.

    :param boost_strength:
        A boost strength of 0.0 has no effect on x.

    :return:
        A tensor representing the activity of x after k-winner take all.
    """
    k = tf.convert_to_tensor(k, dtype=tf.int32)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    boost_strength = tf.math.maximum(boost_strength, 0.0)
    input_shape = tf.shape(x)
    batch_size = input_shape[0]
    n = tf.reduce_prod(x.shape[1:])
    target_density = tf.cast(k, tf.float32) / tf.cast(n, tf.float32)
    boost_factors = tf.exp((target_density - duty_cycles) * boost_strength)
    boosted = x * boost_factors

    # Take the boosted version of the input x, find the top k winners.
    # Compute an output that contains the values of x corresponding to the top k
    # boosted values
    boosted = tf.reshape(boosted, [batch_size, -1])
    flat_x = tf.reshape(x, [batch_size, -1])
    _, indices = tf.math.top_k(input=boosted, k=k, sorted=False)
    mask = tf.reduce_sum(tf.one_hot(indices, tf.shape(boosted)[-1]), axis=1)
    return tf.reshape(mask * flat_x, input_shape)


class KWinnersBase(keras.layers.Layer, metaclass=abc.ABCMeta):
    """
    Base KWinners class.

    :param percent_on:
        The activity of the top k = percent_on * number of input units will be
        allowed to remain, the rest are set to zero.
    :type percent_on: float

    :param k_inference_factor:
        During inference (training=False) we increase percent_on by this factor.
        percent_on * k_inference_factor must be strictly less than 1.0, ideally
        much lower than 1.0
    :type k_inference_factor: float

    :param boost_strength:
        boost strength (0.0 implies no boosting). Must be >= 0.0
    :type boost_strength: float

    :param boost_strength_factor:
        Boost strength factor to use [0..1]
    :type boost_strength_factor: float

    :param duty_cycle_period:
        The period used to calculate duty cycles
    :type duty_cycle_period: int

    :param kwargs:
        Additional args passed to :class:`keras.layers.Layer`
    :type kwargs: dict
    """

    def __init__(
        self,
        percent_on,
        k_inference_factor,
        boost_strength,
        boost_strength_factor,
        duty_cycle_period,
        name=None,
        **kwargs,
    ):
        super(KWinnersBase, self).__init__(name=name, **kwargs)
        self.percent_on = percent_on
        self.percent_on_inference = percent_on * k_inference_factor
        self.k_inference_factor = k_inference_factor
        self.learning_iterations = None
        self.n = 0
        self.k = 0
        self.k_inference = 0

        # Boosting related parameters
        self.boost_strength = None
        self.initial_boost_strength = boost_strength
        self.boost_strength_factor = boost_strength_factor
        self.duty_cycle_period = duty_cycle_period
        self.duty_cycles = None

    def build(self, input_shape):
        super(KWinnersBase, self).build(input_shape=input_shape)

        self.learning_iterations = self.add_variable(
            name="learning_iterations",
            shape=[],
            initializer="zeros",
            trainable=False,
            dtype=tf.int32,
        )
        self.boost_strength = self.add_variable(
            name="boost_strength",
            shape=[],
            initializer=keras.initializers.Constant(self.initial_boost_strength),
            trainable=False,
            dtype=tf.float32,
        )

    def get_config(self):
        config = {
            "percent_on": self.percent_on,
            "k_inference_factor": self.k_inference_factor,
            "boost_strength": K.get_value(self.boost_strength),
            "boost_strength_factor": self.boost_strength_factor,
            "duty_cycle_period": self.duty_cycle_period,
        }
        config.update(super(KWinnersBase, self).get_config())
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    @abc.abstractmethod
    def compute_duty_cycle(self, x):
        r"""
        Compute our duty cycle estimates with the new value. Duty cycles are
        updated according to the following formula:

        .. math::
            dutyCycle = \frac{dutyCycle \times \left( period - batchSize \right)
                                + newValue}{period}

        :param x:
            Current activity of each unit
        """
        raise NotImplementedError

    def update_boost_strength(self):
        """
        Update boost strength using given strength factor during training.
        """
        factor = K.in_train_phase(self.boost_strength_factor, 1.0)
        self.add_update(
            self.boost_strength.assign(self.boost_strength * factor, read_value=False)
        )

    def call(self, inputs, **kwargs):
        inputs = super().call(inputs, **kwargs)
        # FIXME: In eager mode, predict_on_batch method is called with list of inputs
        if isinstance(inputs, list) and len(inputs) == 1:
            inputs = inputs[0]
        return inputs


class KWinners2d(KWinnersBase):
    """
    Applies K-Winner function to Conv2D input tensor.

    :param percent_on:
        The activity of the top k = percent_on * number of input units will be
        allowed to remain, the rest are set to zero.
    :type percent_on: float

    :param k_inference_factor:
        During inference (training=False) we increase percent_on by this factor.
        percent_on * k_inference_factor must be strictly less than 1.0, ideally much
        lower than 1.0
    :type k_inference_factor: float

    :param boost_strength:
        boost strength (0.0 implies no boosting).
    :type boost_strength: float

    :param boost_strength_factor:
        Boost strength factor to use [0..1]
    :type boost_strength_factor: float

    :param duty_cycle_period:
        The period used to calculate duty cycles
    :type duty_cycle_period: int

    :param data_format:
        one of `channels_first` or `channels_last`. The ordering of
        the dimensions in the inputs. `channels_last` corresponds to inputs with
        shape `(batch, height, width, channels)` while `channels_first` corresponds
        to inputs with shape `(batch, channels, height, width)`.
        Similar to `data_format` argument in :class:`keras.layers.Conv2D`.
    :type data_format: str

    :param kwargs:
        Additional args passed to :class:`keras.layers.Layer`
    :type kwargs: dict
    """

    def __init__(
        self,
        percent_on=0.1,
        k_inference_factor=1.5,
        boost_strength=1.0,
        boost_strength_factor=0.9,
        duty_cycle_period=1000,
        data_format=IMAGE_DATA_FORMAT,
        name=None,
        **kwargs,
    ):
        super(KWinners2d, self).__init__(
            percent_on=percent_on,
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
            name=name,
            **kwargs,
        )
        self.data_format = data_format
        if self.data_format == "channels_first":
            # (batch, channels, height, width)
            self.channel_axis = 1
            self.height_axis = 2
            self.width_axis = 3
        else:
            # (batch, height, width, channels)
            self.height_axis = 1
            self.width_axis = 2
            self.channel_axis = 3

        # Not know until `build`
        self.channels = None
        self.scale_factor = None

    def build(self, input_shape):
        super(KWinners2d, self).build(input_shape=input_shape)
        self.channels = input_shape[self.channel_axis]

        duty_cycles_shape = [1] * 4
        duty_cycles_shape[self.channel_axis] = self.channels
        self.duty_cycles = self.add_variable(
            name="duty_cycles",
            shape=duty_cycles_shape,
            initializer=tf.zeros_initializer,
            trainable=False,
            dtype=tf.float32,
        )
        self.n = int(np.prod(input_shape[1:]))
        self.k = int(round(self.n * self.percent_on))
        self.k_inference = int(round(self.k * self.k_inference_factor))

        shape = input_shape.as_list()
        del shape[self.channel_axis]  # Remove channel dim
        del shape[0]  # Remove batch dim
        self.scale_factor = float(np.prod(shape))

    def get_config(self):
        config = {"data_format": self.data_format}
        config.update(super(KWinners2d, self).get_config())
        return config

    def compute_duty_cycle(self, x):
        batch_size = K.shape(x)[0]
        learning_iterations = self.learning_iterations + batch_size
        period = tf.minimum(self.duty_cycle_period, learning_iterations)

        # Scale all dims but the channel dim
        axis = [0, self.height_axis, self.width_axis]
        count = tf.reduce_sum(tf.cast(x > 0, tf.float32), axis=axis) / self.scale_factor
        duty_cycles = self.duty_cycles * tf.cast(period - batch_size, tf.float32)

        # Flatten and add sum
        duty_cycles = tf.reshape(duty_cycles, [-1]) + count

        # Restore duty_cycles shape before dividing
        duty_cycles = tf.reshape(duty_cycles, self.duty_cycles.shape)
        return duty_cycles / tf.cast(period, tf.float32)

    def call(self, inputs, training=None, **kwargs):
        inputs = super().call(inputs, **kwargs)
        k = K.in_test_phase(x=self.k_inference, alt=self.k, training=training)
        kwinners = compute_kwinners(
            x=inputs,
            k=k,
            duty_cycles=self.duty_cycles,
            boost_strength=self.boost_strength,
        )

        duty_cycles = K.in_train_phase(
            lambda: self.compute_duty_cycle(kwinners),
            self.duty_cycles,
            training=training,
        )
        self.add_update(self.duty_cycles.assign(duty_cycles, read_value=False))
        increment = K.in_train_phase(K.shape(inputs)[0], 0, training=training)
        self.add_update(
            self.learning_iterations.assign_add(increment, read_value=False)
        )

        return kwinners


class KWinners(KWinnersBase):
    """Applies K-Winner function to the input tensor.

    :param percent_on:
      The activity of the top k = percent_on * n will be allowed to remain, the
      rest are set to zero.
    :type percent_on: float

    :param k_inference_factor:
      During inference (training=False) we increase percent_on by this factor.
      percent_on * k_inference_factor must be strictly less than 1.0, ideally much
      lower than 1.0
    :type k_inference_factor: float

    :param boost_strength:
      boost strength (0.0 implies no boosting).
    :type boost_strength: float

    :param boost_strength_factor:
      Boost strength factor to use [0..1]
    :type boost_strength_factor: float

    :param duty_cycle_period:
      The period used to calculate duty cycles
    :type duty_cycle_period: int

    :param kwargs:
      Additional args passed to :class:`keras.layers.Layer`
    :type kwargs: dict
    """

    def __init__(
        self,
        percent_on,
        k_inference_factor=1.0,
        boost_strength=1.0,
        boost_strength_factor=1.0,
        duty_cycle_period=1000,
        name=None,
        **kwargs,
    ):
        super(KWinners, self).__init__(
            percent_on=percent_on,
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
            name=name,
            **kwargs,
        )

    def build(self, input_shape):
        super(KWinners, self).build(input_shape=input_shape)
        self.n = int(input_shape[-1])
        self.k = int(round(self.n * self.percent_on))
        self.k_inference = int(round(self.k * self.k_inference_factor))

        self.duty_cycles = self.add_variable(
            name="duty_cycles",
            shape=[self.n],
            initializer=tf.zeros_initializer,
            trainable=False,
            dtype=tf.float32,
        )

    def compute_duty_cycle(self, inputs):
        r"""
        .. math::
            dutyCycle = \frac{dutyCycle \times \left( period - batchSize \right)
                        + newValue}{period}
        """
        batch_size = tf.shape(inputs)[0]
        learning_iterations = self.learning_iterations + batch_size
        period = tf.minimum(self.duty_cycle_period, learning_iterations)
        count = tf.reduce_sum(tf.cast(inputs > 0, tf.float32), axis=0)
        result = self.duty_cycles * tf.cast(period - batch_size, tf.float32) + count
        return result / tf.cast(period, tf.float32)

    def call(self, inputs, training=None, **kwargs):
        inputs = super().call(inputs, **kwargs)
        k = K.in_test_phase(x=self.k_inference, alt=self.k, training=training)
        kwinners = compute_kwinners(
            x=inputs,
            k=k,
            duty_cycles=self.duty_cycles,
            boost_strength=K.get_value(self.boost_strength),
        )

        duty_cycles = K.in_train_phase(
            lambda: self.compute_duty_cycle(kwinners),
            self.duty_cycles,
            training=training,
        )
        self.add_update(self.duty_cycles.assign(duty_cycles, read_value=False))

        increment = K.in_train_phase(K.shape(inputs)[0], 0, training=training)
        self.add_update(
            self.learning_iterations.assign_add(increment, read_value=False)
        )

        return kwinners
