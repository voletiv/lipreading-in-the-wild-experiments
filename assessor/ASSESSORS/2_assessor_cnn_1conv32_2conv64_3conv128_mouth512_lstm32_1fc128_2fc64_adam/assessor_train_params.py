import os

from assessor_functions import *
from assessor_params import *

######################################################
# EXPERIMENT NUMBER
######################################################

experiment_number = 2

######################################################
# PARAMS
######################################################

data_dir = LRW_DATA_DIR

batch_size = 32

train_collect_type = "val"

val_collect_type = "test"

shuffle = True

random_crop = True

verbose = False

# Assessor
mouth_nn = 'cnn'
conv_f_1 = 16
conv_f_2 = 32
conv_f_3 = 64
mouth_features_dim = 512
lstm_units_1 = 32
dense_fc_1 = 128
dense_fc_2 = 64

# Compile
optimizer = 'adam'
loss = 'binary_crossentropy'

# Train
train_lrw_word_set_num_txt_file_names = read_lrw_word_set_num_file_names(collect_type=train_collect_type, collect_by='sample')
train_steps_per_epoch = len(train_lrw_word_set_num_txt_file_names) // batch_size

n_epochs = 100

# Val
val_lrw_word_set_num_txt_file_names = read_lrw_word_set_num_file_names(collect_type=val_collect_type, collect_by='sample')
# val_steps_per_epoch = len(val_lrw_word_set_num_txt_file_names) // batch_size
val_steps_per_epoch = 10     # Set less value so as not to take too much time computing on full val set

# Class weights
# The lipreader is correct 70% of the time
class_weight = {0: .3, 1: .7}

######################################################
# THIS MODEL
######################################################

# THIS MODEL NAME
this_assessor_model = str(experiment_number) + "_assessor_" + mouth_nn + "_mouth" + str(512) + "_lstm" + str(lstm_units_1) + \
                      "_1fc" + str(dense_fc_1) + "_2fc" + str(dense_fc_2) + "_" + optimizer

# Save
this_assessor_save_dir = os.path.realpath(os.path.join(ASSESSOR_SAVE_DIR, this_assessor_model))
