import os

from assessor_functions import *
from assessor_params import *

######################################################
# EXPERIMENT NUMBER
######################################################

ASSESSOR_SAVE_DIR

prev_experiment_number = int(sorted(glob.glob(os.path.join(ASSESSOR_SAVE_DIR, "[0-9]*/")))[-1].split('/')[-2].split('_')[0])

experiment_number = prev_experiment_number + 1
print("Experiment number:", experiment_number)


######################################################
# PARAMS
######################################################

# Data
data_dir = LRW_DATA_DIR

batch_size = 250

train_collect_type = "val"

val_collect_type = "test"

shuffle = True

grayscale_images=True

random_crop = True

random_flip=True

verbose = False

# Assessor
mouth_nn = 'flatten'
conv_f_1 = 16   # doesn't matter
conv_f_2 = 8    # doesn't matter
conv_f_3 = 4    # doesn't matter
mouth_features_dim = 512    # doesn't matter
lstm_units_1 = 4
dense_fc_1 = 64
dense_fc_2 = 32
dropout_p = 0.2

# Compile
optimizer = 'adam'
loss = 'binary_crossentropy'

# Train
train_lrw_word_set_num_txt_file_names = read_lrw_word_set_num_file_names(collect_type=train_collect_type, collect_by='sample')
# train_steps_per_epoch = len(train_lrw_word_set_num_txt_file_names) // batch_size
train_steps_per_epoch = 20     # Set less value so as not to take too much time computing on full train set

n_epochs = 100

# Val
val_lrw_word_set_num_txt_file_names = read_lrw_word_set_num_file_names(collect_type=val_collect_type, collect_by='sample')
# val_steps_per_epoch = len(val_lrw_word_set_num_txt_file_names) // batch_size
val_steps_per_epoch = train_steps_per_epoch     # Set less value so as not to take too much time computing on full val set

# Class weights
# The lipreader is correct 70% of the time
class_weight = {0: .3, 1: .7}

######################################################
# THIS MODEL
######################################################

# THIS MODEL NAME
this_assessor_model = str(experiment_number) + "_assessor_" + mouth_nn

if mouth_nn == 'cnn':
    this_assessor_model = this_assessor_model + '_1conv' + str(conv_f_1) + '_2conv' + str(conv_f_2) + '_3conv' + str(conv_f_3) + "_mouth" + str(mouth_features_dim)

if 'resnet' in mouth_nn:
    this_assessor_model += "_mouth" + str(mouth_features_dim)

this_assessor_model = this_assessor_model + "_lstm" + str(lstm_units_1) + \
                      "_1fc" + str(dense_fc_1) + "_bn_dp" + str(dropout_p) + "_2fc" + str(dense_fc_2) + "_bn_dp" + str(dropout_p) + "_" + optimizer
print("this_assessor_model:", this_assessor_model)

# Save
this_assessor_save_dir = os.path.realpath(os.path.join(ASSESSOR_SAVE_DIR, this_assessor_model))
print("this_assessor_save_dir:", this_assessor_save_dir)
