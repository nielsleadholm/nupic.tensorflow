#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
from nupic.tensorflow.constraints import SparseWeights
from nupic.tensorflow.layers import KWinners, KWinners2d
from tensorflow import keras
from tensorflow.python.keras import backend as K


class GSCSparseCNN(keras.Sequential):
    """Sparse CNN model used to classify `Google Speech Commands` dataset as
    described in `How Can We Be So Dense?`_ paper.

    .. _`How Can We Be So Dense?`: https://arxiv.org/abs/1903.11257

    :param cnn_out_channels: output channels for each CNN layer
    :param cnn_percent_on: Percent of units allowed to remain on each convolution
                           layer
    :param linear_units: Number of units in the linear layer
    :param linear_percent_on: Percent of units allowed to remain on the linear
                              layer
    :param linear_weight_sparsity: Percent of weights that are allowed to be
                                   non-zero in the linear layer
    :param k_inference_factor: During inference (training=False) we increase
                               `percent_on` in all sparse layers by this factor
    :param boost_strength: boost strength (0.0 implies no boosting)
    :param boost_strength_factor: Boost strength factor to use [0..1]
    :param duty_cycle_period: The period used to calculate duty cycles
    :param data_format:
        one of `channels_first` or `channels_last`. The ordering of
        the dimensions in the inputs. `channels_last` corresponds to inputs with
        shape `(batch, height, width, channels)` while `channels_first` corresponds
        to inputs with shape `(batch, channels, height, width)`.
        Similar to `data_format` argument in :class:`keras.layers.Conv2D`.
    """

    def __init__(
        self,
        cnn_out_channels=(64, 64),
        cnn_percent_on=(0.095, 0.125),
        linear_units=1000,
        linear_percent_on=0.1,
        linear_weight_sparsity=0.4,
        boost_strength=1.5,
        boost_strength_factor=0.9,
        k_inference_factor=1.5,
        duty_cycle_period=1000,
        data_format=K.image_data_format(),
        **kwargs,
    ):
        super(GSCSparseCNN, self).__init__(**kwargs)

        if data_format == "channels_first":
            axis = 1
            input_shape = (1, 32, 32)
        else:
            axis = -1
            input_shape = (32, 32, 1)

        self.add(
            keras.layers.Conv2D(
                name="cnn1",
                data_format=data_format,
                input_shape=input_shape,
                filters=cnn_out_channels[0],
                kernel_size=5,
            )
        )
        self.add(keras.layers.BatchNormalization(name="cnn1_batchnorm",
                                                 axis=axis,
                                                 epsilon=1e-05,
                                                 momentum=0.9,
                                                 center=False,
                                                 scale=False))
        self.add(keras.layers.MaxPool2D(name="cnn1_maxpool",
                                        pool_size=2,
                                        padding="same",
                                        data_format=data_format))
        self.add(
            KWinners2d(
                name="cnn1_kwinner",
                data_format=data_format,
                percent_on=cnn_percent_on[0],
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
            )
        )
        self.add(
            keras.layers.Conv2D(
                name="cnn2",
                data_format=data_format,
                filters=cnn_out_channels[1],
                kernel_size=5,
            )
        )
        self.add(keras.layers.BatchNormalization(name="cnn2_batchnorm",
                                                 axis=axis,
                                                 epsilon=1e-05,
                                                 momentum=0.9,
                                                 center=False,
                                                 scale=False))
        self.add(keras.layers.MaxPool2D(name="cnn2_maxpool",
                                        pool_size=2,
                                        padding="same",
                                        data_format=data_format))
        self.add(
            KWinners2d(
                name="cnn2_kwinner",
                data_format=data_format,
                percent_on=cnn_percent_on[1],
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
            )
        )
        self.add(keras.layers.Flatten(name="flatten", data_format=data_format))
        self.add(
            keras.layers.Dense(
                name="linear",
                units=linear_units,
                kernel_constraint=SparseWeights(linear_weight_sparsity),
            )
        )
        self.add(keras.layers.BatchNormalization(name="linear_bn",
                                                 epsilon=1e-05,
                                                 momentum=0.9,
                                                 center=False,
                                                 scale=False))
        self.add(
            KWinners(
                name="linear_kwinner",
                percent_on=linear_percent_on,
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
            )
        )
        self.add(keras.layers.Dense(name="output", units=12))
        self.add(keras.layers.Softmax(axis=1))


class GSCSuperSparseCNN(GSCSparseCNN):
    """Super Sparse CNN model used to classify `Google Speech Commands`
    dataset as described in `How Can We Be So Dense?`_ paper.
    This model provides a sparser version of :class:`GSCSparseCNN`

    .. _`How Can We Be So Dense?`: https://arxiv.org/abs/1903.11257

    """

    def __init__(self, data_format=K.image_data_format()):
        super(GSCSuperSparseCNN, self).__init__(
            linear_units=1500,
            linear_percent_on=0.067,
            linear_weight_sparsity=0.1,
            data_format=data_format,
        )
