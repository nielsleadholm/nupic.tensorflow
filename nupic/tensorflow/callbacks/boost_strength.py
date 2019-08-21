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
from tensorflow.python.keras.callbacks import Callback

from nupic.tensorflow.layers.k_winners import KWinnersBase


class UpdateBoostStrength(Callback):
    """
    Callback used to update KWinner modules boost strength after each epoch.
    """

    def __init__(self):
        super(UpdateBoostStrength, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:
            if isinstance(layer, KWinnersBase):
                layer.update_boost_strength()
