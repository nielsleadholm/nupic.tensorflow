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

import random

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test

from nupic.tensorflow.constraints import SparseWeights
from nupic.tensorflow.layers import KWinners, KWinners2d


# tf.enable_eager_execution()

SEED = 18
CUSTOM_OBJECTS = {
    "KWinners": KWinners,
    "KWinners2d": KWinners2d,
    "SparseWeights": SparseWeights,
}


@tf_test_util.run_all_in_graph_and_eager_modes
class KWinnersTest(test.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.set_random_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

    def setUp(self):
        super(KWinnersTest, self).setUp()
        self.x = np.array(
            [[1.0, 1.2, 1.1, 1.3, 1.0, 1.5, 1.0],
             [1.1, 1.0, 1.2, 1.0, 1.3, 1.0, 1.2]],
            dtype=np.float32
        )
        self.duty_cycle = tf.constant(1.0 / 3.0, shape=(2, 7))

    def test_one(self):
        """boost factor 0, k=3, batch size 2"""

        expected = np.zeros(self.x.shape, dtype=np.float32)
        expected[0, 1] = 1.2
        expected[0, 3] = 1.3
        expected[0, 5] = 1.5
        expected[1, 2] = 1.2
        expected[1, 4] = 1.3
        expected[1, 6] = 1.2
        n = np.prod(self.x.shape[1:])
        kwargs = {
            "percent_on": 3 / n,
            "boost_strength": 0.0}
        with self.cached_session(), keras.utils.custom_object_scope(CUSTOM_OBJECTS):
            testing_utils.layer_test(KWinners,
                                     kwargs=kwargs,
                                     input_data=self.x,
                                     expected_output=expected)


# @test_util.run_all_in_graph_and_eager_modes
class KWinners2dTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.set_random_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

    def setUp(self):
        super().setUp()
        # Batch size 1
        self.x = np.ones((1, 3, 2, 2), dtype=np.float32)
        self.x[0, 0, 1, 0] = 1.1
        self.x[0, 0, 1, 1] = 1.2
        self.x[0, 1, 0, 1] = 1.2
        self.x[0, 2, 1, 0] = 1.3

        # Batch size 2
        self.x2 = np.ones((2, 3, 2, 2), dtype=np.float32)
        self.x2[0, 0, 1, 0] = 1.1
        self.x2[0, 0, 1, 1] = 1.2
        self.x2[0, 1, 0, 1] = 1.2
        self.x2[0, 2, 1, 0] = 1.3

        self.x2[1, 0, 0, 0] = 1.4
        self.x2[1, 1, 0, 0] = 1.5
        self.x2[1, 1, 0, 1] = 1.6
        self.x2[1, 2, 1, 1] = 1.7

    def test_one(self):
        """Equal duty cycle, boost factor 0, k=4, batch size 1."""
        x = self.x
        expected = np.zeros_like(x)
        expected[0, 0, 1, 0] = 1.1
        expected[0, 0, 1, 1] = 1.2
        expected[0, 1, 0, 1] = 1.2
        expected[0, 2, 1, 0] = 1.3

        n = np.prod(x.shape[1:])
        kwargs = {
            "percent_on": 4 / n,
            "k_inference_factor": 1.0,
            "boost_strength": 0.0}
        with self.cached_session(), keras.utils.custom_object_scope(CUSTOM_OBJECTS):
            testing_utils.layer_test(KWinners2d,
                                     kwargs=kwargs,
                                     input_data=x,
                                     expected_output=expected)

    def test_two(self):
        """Equal duty cycle, boost factor 0, k=3."""
        x = self.x
        expected = np.zeros(x.shape)
        expected[0, 0, 1, 1] = 1.2
        expected[0, 1, 0, 1] = 1.2
        expected[0, 2, 1, 0] = 1.3
        n = np.prod(x.shape[1:])
        kwargs = {
            "percent_on": 3 / n,
            "k_inference_factor": 1.0,
            "boost_strength": 0.0}
        with self.cached_session(), keras.utils.custom_object_scope(CUSTOM_OBJECTS):
            testing_utils.layer_test(KWinners2d,
                                     kwargs=kwargs,
                                     input_data=x,
                                     expected_output=expected)

    def test_three(self):
        """Equal duty cycle, boost factor=0, k=4, batch size=2."""
        x = self.x2
        expected = np.zeros(x.shape)
        expected[0, 0, 1, 0] = 1.1
        expected[0, 0, 1, 1] = 1.2
        expected[0, 1, 0, 1] = 1.2
        expected[0, 2, 1, 0] = 1.3
        expected[1, 0, 0, 0] = 1.4
        expected[1, 1, 0, 0] = 1.5
        expected[1, 1, 0, 1] = 1.6
        expected[1, 2, 1, 1] = 1.7

        n = np.prod(x.shape[1:])
        kwargs = {
            "percent_on": 4 / n,
            "k_inference_factor": 1.0,
            "boost_strength": 0.0}
        with self.cached_session(), keras.utils.custom_object_scope(CUSTOM_OBJECTS):
            testing_utils.layer_test(KWinners2d,
                                     kwargs=kwargs,
                                     input_data=x,
                                     expected_output=expected)

    def test_four(self):
        """Equal duty cycle, boost factor=0, k=3, batch size=2."""
        x = self.x2
        expected = np.zeros(x.shape)
        expected[0, 0, 1, 1] = 1.2
        expected[0, 1, 0, 1] = 1.2
        expected[0, 2, 1, 0] = 1.3
        expected[1, 1, 0, 0] = 1.5
        expected[1, 1, 0, 1] = 1.6
        expected[1, 2, 1, 1] = 1.7

        n = np.prod(x.shape[1:])
        kwargs = {
            "percent_on": 3 / n,
            "k_inference_factor": 1.0,
            "boost_strength": 0.0}
        with self.cached_session(), keras.utils.custom_object_scope(CUSTOM_OBJECTS):
            testing_utils.layer_test(KWinners2d,
                                     kwargs=kwargs,
                                     input_data=x,
                                     expected_output=expected)


if __name__ == "__main__":
    test.main()
