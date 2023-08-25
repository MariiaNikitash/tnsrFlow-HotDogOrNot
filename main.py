import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf

from tensorflow.keras import datasets, layers, models

import tensorflow_datasets as tfds



@inproceedings{bossard14,
  title = {Food-101 -- Mining Discriminative Components with Random Forests},
  author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision},
  year = {2014}
}


# TensorFlow has the food 101 dataset already. 
# We will use this! https://www.tensorflow.org/datasets/catalog/food101

# Hot dog is label 55