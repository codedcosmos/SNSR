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
import numpy as np

import model as mg

import os
import random
import math
import glob
from PIL import Image
from pathlib import Path
import random



# Load every image file in dataset
image_paths = list(Path("dataset_prep").rglob("*.[pPjJ][nNpP][gG]"))

# Shuffle
random.shuffle(image_paths)


# Custom Data generator for keras
class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self, image_paths):
        # Set local class variable
        self.batch_size = 1
        self.image_paths = image_paths

        # Prepare
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths)/self.batch_size))

    def __getitem__(self, index):
        # Find list of IDs
        image_paths_temp = self.image_paths[index]

        # Generate data
        X, Y = self.__data_generation(image_paths_temp, index)

        return X, Y

    def on_epoch_end(self):
        # Reset indexes after each epoch
        self.indexes = np.arange(len(self.image_paths))

    def __data_generation(self, image_paths_temp, index):
        try:
            # Normal
            image_raw = Image.open(image_paths_temp)
            width, height = image_raw.size

            # Resize until it's less than or equal to 720p
            while width > 768 or height > 768:
                width, height = round(width/2), round(height/2)

            # Downscaled
            width_ds, height_ds = round(width/2), round(height/2)
            image_raw_ds = image_raw.resize((width_ds, height_ds), Image.ANTIALIAS)

            # Resize so that it's exactly double
            width, height = width_ds*2, height_ds*2
            image_raw = image_raw.resize((width, height), Image.ANTIALIAS)


            # Make numpy
            X = np.asarray(image_raw_ds, dtype=np.float64)
            Y = np.asarray(image_raw, dtype=np.float64)

            # Normalise
            X = X / 255
            Y = Y / 255

            # Add Noise
            noise_z = random.randint(1, 20)

            if noise_z < 12:
                def add_gaussian_noise(image):
                    # image must be scaled in [0, 1]
                    with tf.name_scope('Add_gaussian_noise'):
                        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=(50) / (255), dtype=tf.float32)
                        noise_img = image + (noise / noise_z)
                        noise_img = tf.clip_by_value(noise_img, 0.0, 1.0)
                    return noise_img
                X = add_gaussian_noise(X)

            # Expand
            X = np.expand_dims(X, axis=0)
            Y = np.expand_dims(Y, axis=0)

            # Make sure it's x by y by z by w
            if len(X.shape) < 4 or len(Y.shape) < 4:
                print("")
                print("Shape error for image, <4 - " + str(image_paths_temp) + ", skipping")
                return self.__getitem__((index+1)  % self.__len__())

            # Make sure it's x by y by z by w
            if X.shape[0] != 1 or Y.shape[0] != 1 or X.shape[3] != 3 or Y.shape[3] != 3:
                print("")
                print("Shape error for image, 0-3 - " + str(image_paths_temp) + ", skipping")
                return self.__getitem__((index+1)  % self.__len__())

            return X, Y
        except Exception as e:
            print("")
            print("Encounted error loading image" + str(image_paths_temp) + ", skipping")
            print(e)
            return self.__getitem__((index+1)  % self.__len__())


model = mg.generate_model()
optimizer=tf.keras.optimizers.Adam(1e-4)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

# Variables
NUM_EPOCHS = 10
SIZE_PER_FIT = 5000
FITS_PER_CHECKPOINT = 1

# Split
def chunk(lst, n):
    # Yield successive n-sized chunks from list
    def gen(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    return list(gen(lst, n))


# Checkpoints
checkpoint_dir = 'training_checkpoints_old'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=optimizer, model=model)

# Restore latest
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

c = 0

for i in range(NUM_EPOCHS):

    # Shuffle
    random.shuffle(image_paths)

    # Chunk
    training_data = chunk(image_paths, SIZE_PER_FIT)

    for o in range(0, len(training_data)-1):
        try:
            print()
            print("Training - EPOCH: " + str(i) + "  -  ckpt: " + str(o) + "/" + str(len(training_data)))

            # Create generator
            generator = CustomDataGenerator(training_data[o])

            # Increment checkpoint
            c = c + 1

            # Fit
            model.fit(generator, workers=8, epochs=1)

            # Checkpoint
            if c >= FITS_PER_CHECKPOINT:
                c = 0

                # Save weights
                checkpoint.save(file_prefix=checkpoint_prefix)
                print("Saved checkpoint (ckpt)")

        except Exception as e:
            print("Caught exception")
            print(e)

    # Save weights
    checkpoint.save(file_prefix=checkpoint_prefix)
    print("Saved checkpoint (epoch)")