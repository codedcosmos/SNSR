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
import glob
from PIL import Image
from pathlib import Path

# INPUT IMAGE
INPUT_IMAGE = "INSERT_INPUT_HERE.jpg"
OUTPUT_IMAGE = "INSERT_OUTPUT_HERE.jpg"

# Model
model = mg.generate_model()
optimizer=tf.keras.optimizers.Adam(1e-4)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.Huber(),
              metrics=['accuracy'])

# Load checkpoint
checkpoint_dir = 'training_checkpoints_old'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=optimizer, model=model)


# Restore latest
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# Load image
image_raw = Image.open(INPUT_IMAGE)
image = np.asarray(image_raw, dtype=np.float64)
image = image / 255
image = np.expand_dims(image, axis=0)

raw_output = model.predict(image)
res = raw_output.shape

# Convert output to image and save
raw_output = np.reshape(raw_output, (res[1], res[2], 3))

raw_output = raw_output * 255
raw_output = raw_output.astype(np.uint8)

raw_output[raw_output > 255] = 255
raw_output[raw_output < 0] = 0

output_image = Image.fromarray(raw_output)
output_image.save(OUTPUT_IMAGE)