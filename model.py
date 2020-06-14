# SNSR by codedcosmos
#
# SNSR is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License 3 as published by
# the Free Software Foundation.
# SNSR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License 3 for more details.
# You should have received a copy of the GNU General Public License 3
# along with SNSR.  If not, see <https://www.gnu.org/licenses/>.

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf

from tensorflow import keras

def generate_model():
    model = keras.Sequential([
        # Input
        keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), use_bias=False, padding='same',
                                     input_shape=(None, None, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        # Detection
        keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), use_bias=False, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), use_bias=False, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), use_bias=False, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), use_bias=False, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), use_bias=False, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), use_bias=False, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        # Upscale 2x
        keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), use_bias=False, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        # Interpolate upscaled result
        keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), use_bias=False, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), use_bias=False, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), use_bias=False, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), use_bias=False, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), use_bias=False, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), use_bias=False, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        # RGB Output
        keras.layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'),
    ])
    return model