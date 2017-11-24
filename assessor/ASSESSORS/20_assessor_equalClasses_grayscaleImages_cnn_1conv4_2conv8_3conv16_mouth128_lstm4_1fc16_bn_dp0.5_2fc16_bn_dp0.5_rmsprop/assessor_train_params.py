import os

from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping

from assessor_functions import *
from assessor_params import *

######################################################
# EXPERIMENT NUMBER
######################################################

prev_experiment_number = sorted([int(d.split('/')[-2].split('_')[0]) for d in glob.glob(os.path.join(ASSESSOR_SAVE_DIR, "[0-9]*/"))])[-1]

experiment_number = prev_experiment_number + 1
print("Experiment number:", experiment_number)


######################################################
# PARAMS
######################################################

# BATCH SIZE
batch_size = 250

# Data
data_dir = LRW_DATA_DIR

# Train
train_collect_type = "val"

# Val
val_collect_type = "test"

# Training
shuffle = True
equal_classes = True
grayscale_images=True
random_crop = True
random_flip=True
verbose = False

# Assessor
use_CNN_LSTM = True
use_head_pose = False
mouth_nn = 'cnn'
conv_f_1 = 4
conv_f_2 = 8
conv_f_3 = 16
mouth_features_dim = 128
lstm_units_1 = 4
dense_fc_1 = 16
dense_fc_2 = 16
dropout_p = 0.5

# Compile
optimizer = 'rmsprop'
loss = 'binary_crossentropy'

# Train
train_lrw_word_set_num_txt_file_names = read_lrw_word_set_num_file_names(collect_type=train_collect_type, collect_by='sample')
train_steps_per_epoch = len(train_lrw_word_set_num_txt_file_names) // batch_size
# train_steps_per_epoch = 20     # Set less value so as not to take too much time computing on full train set

n_epochs = 200

# Val
val_lrw_word_set_num_txt_file_names = read_lrw_word_set_num_file_names(collect_type=val_collect_type, collect_by='sample')
# val_steps_per_epoch = len(val_lrw_word_set_num_txt_file_names) // batch_size
val_steps_per_epoch = train_steps_per_epoch     # Set less value so as not to take too much time computing on full val set

# Class weights
# The lipreader is correct 70% of the time
if equal_classes:
    class_weight = None
else:
    class_weight = {0: .3, 1: .7}

######################################################
# THIS MODEL
######################################################

# THIS MODEL NAME
this_assessor_model = str(experiment_number) + "_assessor"

if equal_classes:
    this_assessor_model += "_equalClasses"

if use_CNN_LSTM:
    if grayscale_images:
        this_assessor_model += "_grayscaleImages"

    this_assessor_model += "_" + mouth_nn

    if mouth_nn == 'cnn':
        this_assessor_model += '_1conv' + str(conv_f_1) + '_2conv' + str(conv_f_2) + '_3conv' + str(conv_f_3) + "_mouth" + str(mouth_features_dim)

    elif 'resnet' in mouth_nn:
        this_assessor_model += "_mouth" + str(mouth_features_dim)

    if use_head_pose:
        this_assessor_model += "_headPose"

    this_assessor_model += "_lstm" + str(lstm_units_1)

this_assessor_model += "_1fc" + str(dense_fc_1) + "_bn_dp" + str(dropout_p) + "_2fc" + str(dense_fc_2) + "_bn_dp" + str(dropout_p) + "_" + optimizer

print("this_assessor_model:", this_assessor_model)

# Save
this_assessor_save_dir = os.path.realpath(os.path.join(ASSESSOR_SAVE_DIR, this_assessor_model))
print("this_assessor_save_dir:", this_assessor_save_dir)

######################################################
# SGD
######################################################

if optimizer == 'sgd':
    optimizer = SGD(lr=0.01, momentum=0.5, decay=0.005)
