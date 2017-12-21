import os

from keras.optimizers import SGD, Adam
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
batch_size = 32

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
random_flip = True
verbose = False
use_softmax = True
use_softmax_ratios = False

# Assessor
mouth_nn = 'syncnet'
# doesn't matter:
my_resnet_repetitions = [1, 1]
use_CNN_LSTM = True
use_head_pose = False
conv_f_1 = 4
conv_f_2 = 8
conv_f_3 = 16
mouth_features_dim = 32
lstm_units_1 = 2
dense_fc_1 = 16
dropout_p1 = 0.2
dense_fc_2 = 16
dropout_p2 = 0.2
individual_dense=False
lr_dense_fc=8
lr_softmax_fc=8

if mouth_nn == 'syncnet':
    # Params
    trainable_syncnet = False
    use_head_pose = True
    lstm_units_1 = 8
    individual_dense = True
    lr_dense_fc = 8
    use_softmax = True
    lr_softmax_fc = 8
    use_softmax_ratios = True
    dense_fc_1 = 8
    dropout_p1 = 0.2
    dense_fc_2 = 16
    dropout_p2 = 0.2
    # Constants
    grayscale_images = True
    use_CNN_LSTM = True
    mouth_features_dim = 128
elif mouth_nn == 'syncnet_preds':
    # Params
    use_head_pose = True
    lstm_units_1 = 8
    individual_dense = True
    lr_dense_fc = 8
    use_softmax = True
    lr_softmax_fc = 8
    use_softmax_ratios = False
    dense_fc_1 = 16
    dropout_p1 = 0.2
    dense_fc_2 = 16
    dropout_p2 = 0.2
    # Constants
    grayscale_images = True
    use_CNN_LSTM = True
    mouth_features_dim = 128
    trainable_syncnet = False

# Use Resnet in the last layer
# last_fc = 'resnet152'
last_fc = None

# Compile
optimizer_name = 'adam'
adam_lr = 1e-3
adam_lr_decay = 1e-3
loss = 'binary_crossentropy'

# Train
train_lrw_word_set_num_txt_file_names = read_lrw_word_set_num_file_names(collect_type=train_collect_type, collect_by='sample')
train_steps_per_epoch = len(train_lrw_word_set_num_txt_file_names) // batch_size
# train_steps_per_epoch = 20     # Set less value so as not to take too much time computing on full train set

n_epochs = 1000

# Val
val_lrw_word_set_num_txt_file_names = read_lrw_word_set_num_file_names(collect_type=val_collect_type, collect_by='sample')
val_steps_per_epoch = len(val_lrw_word_set_num_txt_file_names) // batch_size
# val_steps_per_epoch = train_steps_per_epoch     # Set less value so as not to take too much time computing on full val set

# Class weights
# The lipreader is correct 70% of the time
if equal_classes:
    class_weight = None
else:
    class_weight = {0: .3, 1: .7}

######################################################
# OPTIMIZER
######################################################

if optimizer_name == 'sgd':
    optimizer = SGD(lr=0.01, momentum=0.5, decay=0.005)
elif optimizer_name == 'adam':
    optimizer = Adam(lr=adam_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=adam_lr_decay)
else:
    optimizer = optimizer_name

######################################################
# THIS MODEL
######################################################


def make_this_assessor_model_name_and_save_dir_name(experiment_number, equal_classes, use_CNN_LSTM,
                                                    mouth_nn, trainable_syncnet, grayscale_images,
                                                    conv_f_1, conv_f_2, conv_f_3, mouth_features_dim,
                                                    use_head_pose, lstm_units_1, use_softmax, use_softmax_ratios,
                                                    individual_dense, lr_dense_fc, lr_softmax_fc,
                                                    last_fc, dense_fc_1, dropout_p1, dense_fc_2, dropout_p2,
                                                    optimizer_name, adam_lr=1e-3, adam_lr_decay=1e-3):
    # THIS MODEL NAME
    this_assessor_model_name = str(experiment_number) + "_assessor"

    if equal_classes:
        this_assessor_model_name += "_equalClasses"

    if use_CNN_LSTM:
        if mouth_nn == 'syncnet':
            this_assessor_model_name += "_syncnet"
            if trainable_syncnet:
                this_assessor_model_name += "Trainable"
            else:
                this_assessor_model_name += "Untrainable"
        elif mouth_nn == 'syncnet_preds':
            this_assessor_model_name += "_syncnetPreds"

        else:
            if grayscale_images:
                this_assessor_model_name += "_grayscaleImages"

            this_assessor_model_name += "_" + mouth_nn

            if mouth_nn == 'cnn':
                this_assessor_model_name += '_1conv' + str(conv_f_1) + '_2conv' + str(conv_f_2) + '_3conv' + str(conv_f_3) + "_mouth" + str(mouth_features_dim)

            elif 'resnet' in mouth_nn:
                this_assessor_model_name += "_mouth" + str(mouth_features_dim)

        if use_head_pose:
            this_assessor_model_name += "_headPose"

        this_assessor_model_name += "_lstm" + str(lstm_units_1) + "_nOfFrames_LRdense"

    if individual_dense:
        this_assessor_model_name += "_fc" + str(lr_dense_fc)

    if use_softmax:
        this_assessor_model_name += "_LRsoftmax"
        if individual_dense:
            this_assessor_model_name += "_fc" + str(lr_softmax_fc)

    if use_softmax_ratios:
        this_assessor_model_name += "_LRsoftmaxRatios"

    if last_fc == None:
        this_assessor_model_name += "_1fc" + str(dense_fc_1) + "_bn_dp" + str(dropout_p1) + "_2fc" + str(dense_fc_2) + "_bn_dp" + str(dropout_p2)
    else:
        this_assessor_model_name += "_" + last_fc

    this_assessor_model_name += "_" + optimizer_name
    if optimizer_name == 'adam':
        this_assessor_model_name += "_lr" + str(adam_lr)
        this_assessor_model_name += "_lrDecay" + str(adam_lr_decay)

    print("this_assessor_model_name:", this_assessor_model_name)

    # Save
    this_assessor_save_dir = os.path.realpath(os.path.join(ASSESSOR_SAVE_DIR, this_assessor_model_name))
    print("this_assessor_save_dir:", this_assessor_save_dir)

    # Return
    return this_assessor_model_name, this_assessor_save_dir
