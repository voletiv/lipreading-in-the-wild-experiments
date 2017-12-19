import os
if 'voleti.vikram' in os.getcwd():
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# np.random.seed(29)
tf.set_random_seed(29)

from keras import backend as K
from keras.models import Model, Sequential, model_from_json, model_from_yaml
from keras.layers import Masking, TimeDistributed, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D
from keras.layers import Flatten, Dense, Input, Reshape, Add, Concatenate, LSTM, Dropout
from keras.regularizers import l2
from keras.callbacks import Callback

from assessor_params import *
from resnet import *
from syncnet_functions import *

#########################################################
# MODEL
#########################################################


def my_assessor_model(use_CNN_LSTM=True, use_head_pose=True, mouth_nn='cnn', trainable_syncnet=False,
                      grayscale_images=False, my_resnet_repetitions=[2, 2, 2, 2],
                      conv_f_1=32, conv_f_2=64, conv_f_3=128, mouth_features_dim=512,
                      lstm_units_1=32, use_softmax=True, use_softmax_ratios=False,
                      individual_dense=False, lr_dense_fc=8, lr_softmax_fc=8,
                      dense_fc_1=128, dropout_p1=0.2, dense_fc_2=64, dropout_p2=0.2, last_fc=None):

    if grayscale_images:
        MOUTH_CHANNELS = 1

    if mouth_nn == 'syncnet':
        MOUTH_CHANNELS = 5

    my_input_n_of_frames = Input(shape=(1,), name='n_of_frames')
    my_input_lipreader_dense = Input(shape=(1024,), name='lipreader_middle')

    if use_CNN_LSTM:

        if mouth_nn == 'syncnet_preds':
            my_input_mouth_images = Input(shape=(TIME_STEPS, 128), name='syncnet_preds')
            cnn_features = my_input_mouth_images
        else:
            mouth_input_shape = (MOUTH_H, MOUTH_W, MOUTH_CHANNELS)
            my_input_mouth_images = Input(shape=(TIME_STEPS, *mouth_input_shape), name='mouth_images')
            if mouth_nn == 'cnn':
                mouth_feature_model = my_timedistributed_cnn_model((TIME_STEPS, *mouth_input_shape), conv_f_1, conv_f_2, conv_f_3, mouth_features_dim)
            elif mouth_nn == 'resCNN':
                mouth_feature_model = my_resnet_like_timeDistributed_CNN((TIME_STEPS, *mouth_input_shape), conv_f_1, conv_f_2, conv_f_3, mouth_features_dim)
            elif mouth_nn == 'resnet18':
                mouth_feature_model = ResnetBuilder.build_resnet_18((TIME_STEPS, *mouth_input_shape), mouth_features_dim, time_distributed=True)
            elif mouth_nn == 'resnet34':
                mouth_feature_model = ResnetBuilder.build_resnet_34((TIME_STEPS, *mouth_input_shape), mouth_features_dim, time_distributed=True)
            elif mouth_nn == 'resnet50':
                mouth_feature_model = ResnetBuilder.build_resnet_50((TIME_STEPS, *mouth_input_shape), mouth_features_dim, time_distributed=True)
            elif mouth_nn == 'resnet101':
                mouth_feature_model = ResnetBuilder.build_resnet_101((TIME_STEPS, *mouth_input_shape), mouth_features_dim, time_distributed=True)
            elif mouth_nn == 'resnet152':
                mouth_feature_model = ResnetBuilder.build_resnet_152((TIME_STEPS, *mouth_input_shape), mouth_features_dim, time_distributed=True)
            elif mouth_nn == 'my_resnet':
                mouth_feature_model = ResnetBuilder.build((TIME_STEPS, *mouth_input_shape), mouth_features_dim, basic_block, my_resnet_repetitions, time_distributed=True)
            elif mouth_nn == 'flatten':
                mouth_feature_model = Reshape((TIME_STEPS, -1), input_shape=(TIME_STEPS, *mouth_input_shape))
            elif mouth_nn == 'syncnet':
                mouth_feature_model = TimeDistributed(load_pretrained_syncnet_model(version='v4', mode='lip', verbose=False), input_shape=(TIME_STEPS, *mouth_input_shape), name='syncnet')
                if not trainable_syncnet:
                    mouth_feature_model.layer.trainable = False
            cnn_features = mouth_feature_model(my_input_mouth_images)

        if use_head_pose:
            my_input_head_poses = Input(shape=(TIME_STEPS, 3), name='head_poses')
            lstm_input = Concatenate()([cnn_features, my_input_head_poses])
        else:
            lstm_input = cnn_features

        lstm_output = LSTM(lstm_units_1, activation='tanh', kernel_regularizer=l2(1.e-4), return_sequences=False)(lstm_input)

    if individual_dense:
        d1 = Dense(lr_dense_fc, kernel_regularizer=l2(1.e-4))(my_input_lipreader_dense)
        a1 = Activation('relu', name='relu_lr_dense')(d1)
        lipreader_dense_features = BatchNormalization()(a1)
        if use_softmax:
            my_input_lipreader_softmax = Input(shape=(500,), name='lipreader_softmax')
            d2 = Dense(lr_softmax_fc, kernel_regularizer=l2(1.e-4))(my_input_lipreader_softmax)
            a2 = Activation('relu', name='relu_lr_softmax')(d2)
            lipreader_softmax_features = BatchNormalization()(a2)
    else:
        lipreader_dense_features = my_input_lipreader_dense
        if use_softmax:
            lipreader_softmax_features = my_input_lipreader_softmax

    if use_softmax_ratios:
        my_input_lipreader_softmax_ratios = Input(shape=(2,), name='lipreader_softmax_ratios')
        lipreader_softmax_ratio_features = my_input_lipreader_softmax_ratios

    to_concatenate = []
    if use_CNN_LSTM:
        to_concatenate += [lstm_output]
    to_concatenate += [my_input_n_of_frames, lipreader_dense_features]
    if use_softmax:
        to_concatenate += [lipreader_softmax_features]
    if use_softmax_ratios:
        to_concatenate += [lipreader_softmax_ratio_features]

    concatenated_features = Concatenate()(to_concatenate)


    if last_fc is None:

        fc1 = Dense(dense_fc_1, kernel_regularizer=l2(1.e-4))(concatenated_features)
        ac1 = Activation('relu', name='relu_fc1')(fc1)
        bn1 = BatchNormalization()(ac1)
        dp1 = Dropout(dropout_p1, name='dropout1_p'+str(dropout_p1))(bn1)

        fc2 = Dense(dense_fc_2, kernel_regularizer=l2(1.e-4))(bn1)
        ac2 = Activation('relu', name='relu_fc2')(fc2)
        bn2 = BatchNormalization()(ac2)
        dp2 = Dropout(dropout_p2, name='dropout2_p'+str(dropout_p2))(bn2)

        assessor_output = Dense(1, activation='sigmoid', name='sigmoid')(dp2)

    elif 'resnet' in last_fc:

        if last_fc == 'resnet18':
            last_fc_model = ResnetBuilder.build_resnet_18((int(concatenated_features.shape[1]), 1, 1), 32, time_distributed=False)
        elif last_fc == 'resnet34':
            last_fc_model = ResnetBuilder.build_resnet_34((int(concatenated_features.shape[1]), 1, 1), 32, time_distributed=False)
        elif last_fc == 'resnet50':
            last_fc_model = ResnetBuilder.build_resnet_50((int(concatenated_features.shape[1]), 1, 1), 32, time_distributed=False)
        elif last_fc == 'resnet101':
            last_fc_model = ResnetBuilder.build_resnet_101((int(concatenated_features.shape[1]), 1, 1), 32, time_distributed=False)
        elif last_fc == 'resnet152':
            last_fc_model = ResnetBuilder.build_resnet_152((int(concatenated_features.shape[1]), 1, 1), 32, time_distributed=False)

        fc_input = Reshape((int(concatenated_features.shape[1]), 1, 1), input_shape=(int(concatenated_features.shape[1]),))(concatenated_features)
        resnet_output = last_fc_model(fc_input)

        assessor_output = Dense(1, activation='sigmoid', name='sigmoid')(resnet_output)

    inputs = []
    outputs = assessor_output
    if use_CNN_LSTM:
        inputs += [my_input_mouth_images]
        if use_head_pose:
            inputs += [my_input_head_poses]
    inputs += [my_input_n_of_frames, my_input_lipreader_dense]
    if use_softmax:
        inputs += [my_input_lipreader_softmax]
    if use_softmax_ratios:
        inputs += [my_input_lipreader_softmax_ratios]

    assessor = Model(inputs=inputs, outputs=outputs)

    assessor.summary()

    return assessor


#########################################################
# MY RESNET-like CNN
#########################################################


def my_resnet_like_timeDistributed_CNN(input_shape, conv_f_1, conv_f_2, conv_f_3, cnn_dense_fc_1):

    _handle_dim_ordering()

    model = Sequential()

    # First
    model.add(TimeDistributed(Conv2D(filters=conv_f_1, kernel_size=(7, 7), strides=(2, 2), padding='same', kernel_regularizer=l2(1.e-4)), input_shape=input_shape))
    model.add(TimeDistributed(BatchNormalization(axis=CHANNEL_AXIS)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")))

    # Residual-like block
    model.add(TimeDistributed(Conv2D(filters=conv_f_2, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))))
    model.add(TimeDistributed(BatchNormalization(axis=CHANNEL_AXIS)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv2D(filters=conv_f_2, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(1.e-4))))

    # Residual-like block
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Conv2D(filters=conv_f_3, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(1.e-4))))
    model.add(TimeDistributed(BatchNormalization(axis=CHANNEL_AXIS)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv2D(filters=conv_f_3, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(1.e-4))))

    # End
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))

    # Classifier-like block
    model.add(TimeDistributed(AveragePooling2D(pool_size=(model.output_shape[ROW_AXIS], model.output_shape[COL_AXIS]), strides=(1, 1))))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(units=cnn_dense_fc_1, kernel_initializer="he_normal", activation='relu')))
    model.add(TimeDistributed(BatchNormalization()))

    return model


def my_small_resnet(input_shape, conv_f_1, conv_f_2, conv_f_3, cnn_dense_fc_1):

    _handle_dim_ordering()

    model = Sequential()

    # First
    model.add(TimeDistributed(Conv2D(filters=conv_f_1, kernel_size=(7, 7), strides=(2, 2), padding='same', kernel_regularizer=l2(1.e-4)), input_shape=input_shape))
    model.add(TimeDistributed(BatchNormalization(axis=CHANNEL_AXIS)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")))

    # Residual-like block
    model.add(TimeDistributed(Conv2D(filters=conv_f_2, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))))
    model.add(TimeDistributed(BatchNormalization(axis=CHANNEL_AXIS)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv2D(filters=conv_f_2, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(1.e-4))))
    model.add()

    # Residual-like block
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Conv2D(filters=conv_f_3, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(1.e-4))))
    model.add(TimeDistributed(BatchNormalization(axis=CHANNEL_AXIS)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv2D(filters=conv_f_3, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(1.e-4))))

    # End
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))

    # Classifier-like block
    model.add(TimeDistributed(AveragePooling2D(pool_size=(model.output_shape[ROW_AXIS], model.output_shape[COL_AXIS]), strides=(1, 1))))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(units=cnn_dense_fc_1, kernel_initializer="he_normal", activation='relu')))
    model.add(TimeDistributed(BatchNormalization()))

    return model


def _handle_dim_ordering(time_distributed=True, verbose=False):
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if verbose:
        print("_handle_dim_ordering")
    if time_distributed:
        if K.image_dim_ordering() == 'tf':
            ROW_AXIS = 2
            COL_AXIS = 3
            CHANNEL_AXIS = -1
        else:
            CHANNEL_AXIS = 2
            ROW_AXIS = 3
            COL_AXIS = 4
    else:
        if K.image_dim_ordering() == 'tf':
            ROW_AXIS = 1
            COL_AXIS = 2
            CHANNEL_AXIS = 3
        else:
            CHANNEL_AXIS = 1
            ROW_AXIS = 2
            COL_AXIS = 3


#########################################################
# MY CNN
#########################################################


def my_timedistributed_cnn_model(input_shape, conv_f_1, conv_f_2, conv_f_3, cnn_dense_fc_1, masking=False):

    model = Sequential()

    if masking:
        model.add(Masking(mask_value=0.0, input_shape=input_shape))
        model.add(TimeDistributed(Conv2D(filters=conv_f_1, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(1.e-4))))
    else:
        model.add(TimeDistributed(Conv2D(filters=conv_f_1, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(1.e-4)),
            input_shape=input_shape))

    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')))

    model.add(TimeDistributed(Conv2D(filters=conv_f_2, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=l2(1.e-4))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')))

    model.add(TimeDistributed(Conv2D(filters=conv_f_3, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=l2(1.e-4))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')))

    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(cnn_dense_fc_1, kernel_regularizer=l2(1.e-4))))

    return model


def my_timedistributed_cnn_model_with_skip(input_shape, conv_f_1, conv_f_2, conv_f_3, cnn_dense_fc_1, masking=False):

    time_steps = input_shape[0]
    mouth_shape = (input_shape[1], input_shape[2], input_shape[3])

    my_input = Input(shape=input_shape)

    a_c_out = TimeDistributed(Conv2D(filters=conv_f_1, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(1.e-4)),
                        input_shape=input_shape)(my_input)
    a_bn_out = TimeDistributed(BatchNormalization())(a_c_out)
    a_act_out = TimeDistributed(Activation('relu'))(a_bn_out)
    a_pool_out = TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))(a_act_out)
    # a_


    model.add(TimeDistributed(Conv2D(filters=conv_f_2, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=l2(1.e-4))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')))

    model.add(TimeDistributed(Conv2D(filters=conv_f_3, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=l2(1.e-4))))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')))

    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(cnn_dense_fc_1, kernel_regularizer=l2(1.e-4))))

    return model

def my_cnn_model(input_shape, conv_f_1, conv_f_2, conv_f_3, cnn_dense_fc_1):
    model = Sequential()
    model.add(Conv2D(filters=conv_f_1, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(1.e-4),
        input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Conv2D(filters=conv_f_2, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=l2(1.e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Conv2D(filters=conv_f_3, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=l2(1.e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(cnn_dense_fc_1, kernel_regularizer=l2(1.e-4)))
    return model


def make_time_distributed_simple(model, TIME_STEPS, input_shape):
    time_distributed_model = Sequential()
    for model_layer_index in range(len(model.layers)):
        # print(model_layer_index)
        if model_layer_index == 0:
            time_distributed_model.add(TimeDistributed(model.layers[model_layer_index], input_shape=(TIME_STEPS, *input_shape)))
        else:
            time_distributed_model.add(TimeDistributed(model.layers[model_layer_index]))
    # Return
    return time_distributed_model


#########################################################
# CALLBACK
#########################################################


class CheckpointAndMakePlots(Callback):

    # Init
    def __init__(self, file_name_pre="assessor_cnn_adam", this_assessor_save_dir="."):
        self.file_name_pre = file_name_pre
        self.this_assessor_save_dir = this_assessor_save_dir

    # On train start
    def on_train_begin(self, logs={}):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = 1000

    # At every epoch
    def on_epoch_end(self, epoch, logs={}):

        # Get
        tl = logs.get('loss')
        ta = logs.get('acc')
        vl = logs.get('val_loss')
        va = logs.get('val_acc')
        print("\n", tl, ta, vl, va)

        # Append losses and accs
        self.train_losses.append(tl)
        self.val_losses.append(vl)
        self.train_accuracies.append(ta)
        self.val_accuracies.append(va)

        # Save model
        if vl < self.best_val_loss:
            self.best_val_loss = vl
            self.save_model_checkpoint(epoch, tl, ta, vl, va)

        # Save history
        self.save_history()

        # Plot graphs
        self.plot_and_save_losses_and_accuracies(epoch)

    # Save model checkpoint
    def save_model_checkpoint(self, epoch, tl, ta, vl, va):
        model_file_path = os.path.join(self.this_assessor_save_dir,
            self.file_name_pre + "_epoch{0:03d}_tl{1:.4f}_ta{2:.4f}_vl{3:.4f}_va{4:.4f}.hdf5".format(epoch, tl, ta, vl, va))
        print("Saving model", model_file_path)
        self.model.save_weights(model_file_path)

    def save_history(self):
        print("Saving history in", self.this_assessor_save_dir)
        np.savetxt(os.path.join(self.this_assessor_save_dir, self.file_name_pre + "_loss_history.txt"), self.train_losses, delimiter=",")
        np.savetxt(os.path.join(self.this_assessor_save_dir, self.file_name_pre + "_acc_history.txt"), self.train_accuracies, delimiter=",")
        np.savetxt(os.path.join(self.this_assessor_save_dir, self.file_name_pre + "_val_loss_history.txt"), self.val_losses, delimiter=",")
        np.savetxt(os.path.join(self.this_assessor_save_dir, self.file_name_pre + "_val_acc_history.txt"), self.val_accuracies, delimiter=",")

    # Plot and save losses and accuracies
    def plot_and_save_losses_and_accuracies(self, epoch):
        print("Saving plot for epoch", str(epoch), ":",
            os.path.join(self.this_assessor_save_dir, self.file_name_pre + "_plots.png"))

        plt.subplot(121)
        plt.plot(self.train_losses, label='train_loss')
        plt.plot(self.val_losses, label='val_loss')
        leg = plt.legend(loc='upper right', fontsize=11, fancybox=True)
        leg.get_frame().set_alpha(0.3)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title("Loss")

        plt.subplot(122)
        plt.plot(self.train_accuracies, label='train_acc')
        plt.plot(self.val_accuracies, label='val_acc')
        leg = plt.legend(loc='lower right', fontsize=11, fancybox=True)
        leg.get_frame().set_alpha(0.3)
        plt.xlabel('epochs')
        plt.ylabel('acc')
        plt.yticks(np.arange(0, 1.05, 0.05))
        plt.tick_params(axis='y', which='both',
                        labelleft='on', labelright='on')
        plt.gca().yaxis.grid(True)
        plt.title("Accuracy")

        plt.tight_layout()
        # plt.subplots_adjust(top=0.85)
        plt.suptitle(self.file_name_pre, fontsize=10)
        plt.savefig(os.path.join(self.this_assessor_save_dir,
                                 self.file_name_pre + "_plots_loss_acc.png"))
        plt.close()


#########################################################
# WRITE MODEL ARCHITECTURE
#########################################################


def write_model_architecture(model, file_type='json', file_name="model"):
    if file_type == 'json':
        # serialize model to JSON
        model_json = model.to_json()
        with open(file_name+'.json', "w") as json_file:
            json_file.write(model_json)
    elif file_type == 'yaml':
        # serialize model to YAML
        model_yaml = model.to_yaml()
        with open(file_name+'.yaml', "w") as yaml_file:
            yaml_file.write(model_yaml)
    else:
        print("file_type can only be 'json' or 'yaml'")


#########################################################
# READ MODEL ARCHITECTURE
#########################################################


def read_my_model(model_file_name="model.json", weights_file_name=None):
    # Load model
    print("Loading model from", model_file_name, "...")
    # json
    if model_file_name.split('.')[-1] == 'json':
        with open(model_file_name, 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
    # yaml
    elif model_file_name.split('.')[-1] == 'yaml' or file_name.split('.')[-1] == 'yml':
        with open(model_file_name, 'r') as yaml_file:
            loaded_model_yaml = yaml_file.read()
        loaded_model = model_from_yaml(loaded_model_yaml)
    else:
        print("file_type can only be 'json' or 'yaml'")
    # Load weights
    if weights_file_name is not None:
        print("Loading weights from", weights_file_name, "...")
        loaded_model.load_weights(weights_file_name)
    # Return
    print("Done loading model.")
    return loaded_model

