import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf

from tensorflow.keras import datasets, layers, models

import tensorflow_datasets as tfds



#@inproceedings{bossard14,
#  title = {Food-101 -- Mining Discriminative Components with Random Forests},
#  author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
#  booktitle = {European Conference on Computer Vision},
#  year = {2014}
#}


# TensorFlow has the food 101 dataset already. 
# We will use this! https://www.tensorflow.org/datasets/catalog/food101

# Hot dog is label 55

ds, ds_info = tfds.load('food101', shuffle_files=True, as_supervised=True, with_info=True)

train_ds, valid_ds = ds["train"], ds["validation"]

fig = tfds.show_examples(train_ds, ds_info)


MAX_SIDE_LEN = 128
HOT_DOG_CLASS = 55
train_ds = train_ds.map(
    lambda image, label: (tf.cast(tf.image.resize(image, [MAX_SIDE_LEN, MAX_SIDE_LEN]), dtype=tf.int32),
                                  tf.cast(label == HOT_DOG_CLASS, tf.int32))
)



valid_ds = train_ds.map(
    lambda image, label: (tf.cast(tf.image.resize(image, [MAX_SIDE_LEN, MAX_SIDE_LEN]), dtype=tf.int32),
                                  tf.cast(label == HOT_DOG_CLASS, tf.int32))
)

