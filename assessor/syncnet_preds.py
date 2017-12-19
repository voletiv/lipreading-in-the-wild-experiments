import os
if 'voleti.vikram' in os.getcwd():
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm

np.random.seed(29)
tf.set_random_seed(29)

from keras import backend as K
K.set_learning_phase(False)
from keras.layers import Input, TimeDistributed
from keras.models import Model

from assessor_functions import *
from assessor_params import *
from syncnet_functions import *

mouth_input_shape = (112, 112, 5)

# BATCH SIZE
batch_size = 32

# Data
data_dir = LRW_DATA_DIR

# Train
train_collect_type = "val"

# Val
val_collect_type = "test"

# Training
shuffle = False
equal_classes = False
grayscale_images=True
random_crop = False
random_flip = False
verbose = False
use_softmax = False
get_last_smaller_batch = True

# Assessor
mouth_nn = 'syncnet'
# Params
use_head_pose = False
use_softmax = False
use_softmax_ratios = False
# Constants
grayscale_images = True
use_CNN_LSTM = True

# SYNCNET
syncnet_input = Input(shape=(TIME_STEPS, *mouth_input_shape))
syncnet_output = TimeDistributed(load_pretrained_syncnet_model(version='v4', mode='lip', verbose=False))(syncnet_input)
syncnet_model = Model(inputs=[syncnet_input], outputs=[syncnet_output])

# TRAIN
train_lrw_word_set_num_txt_file_names = read_lrw_word_set_num_file_names(collect_type=train_collect_type, collect_by='sample')
n_batches = len(train_lrw_word_set_num_txt_file_names) // batch_size + 1

train_generator = generate_assessor_data_batches(batch_size=batch_size, data_dir=data_dir, collect_type=train_collect_type, shuffle=shuffle, equal_classes=equal_classes,
                                                 use_CNN_LSTM=use_CNN_LSTM, mouth_nn="syncnet", grayscale_images=grayscale_images, random_crop=random_crop, random_flip=random_flip, use_head_pose=use_head_pose,
                                                 use_softmax=use_softmax, use_softmax_ratios=use_softmax_ratios, verbose=verbose, skip_batches=0, get_last_smaller_batch=get_last_smaller_batch)

# Y
train_Y = np.empty((0, TIME_STEPS, 128))

# Predict
for batch in tqdm.tqdm(range(n_batches)):
    ((X, _, _), _) = next(train_generator)
    y = syncnet_model.predict(X)
    train_Y = np.vstack((train_Y, y))
    if batch % 10 == 0:
        np.save('/shared/fusor/home/voleti.vikram/LRW_val_syncnet_preds', train_Y)

np.save('/shared/fusor/home/voleti.vikram/LRW_val_syncnet_preds', train_Y)


# VAL
val_lrw_word_set_num_txt_file_names = read_lrw_word_set_num_file_names(collect_type=val_collect_type, collect_by='sample')
n_batches = len(val_lrw_word_set_num_txt_file_names) // batch_size + 1

val_generator = generate_assessor_data_batches(batch_size=batch_size, data_dir=data_dir, collect_type=val_collect_type, shuffle=shuffle, equal_classes=equal_classes,
                                               use_CNN_LSTM=use_CNN_LSTM, mouth_nn="syncnet", grayscale_images=grayscale_images, random_crop=random_crop, random_flip=random_flip, use_head_pose=use_head_pose,
                                               use_softmax=use_softmax, use_softmax_ratios=use_softmax_ratios, verbose=verbose, skip_batches=0, get_last_smaller_batch=get_last_smaller_batch)

# Y
val_Y = np.empty((0, TIME_STEPS, 128))

# Predict
for batch in tqdm.tqdm(range(n_batches)):
    ((X, _, _), _) = next(val_generator)
    y = syncnet_model.predict(X)
    val_Y = np.vstack((val_Y, y))
    if batch % 10 == 0:
        np.save('/shared/fusor/home/voleti.vikram/LRW_test_syncnet_preds', val_Y)

np.save('/shared/fusor/home/voleti.vikram/LRW_test_syncnet_preds', val_Y)


