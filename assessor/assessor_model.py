import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Model, Sequential, model_from_json, model_from_yaml
from keras.layers import Masking, TimeDistributed, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.layers import Flatten, Dense, Input, Add, Concatenate, LSTM
from keras.regularizers import l2
from keras.callbacks import Callback

from assessor_params import *
from resnet import ResnetBuilder


#########################################################
# MODEL
#########################################################


def my_assessor_model(mouth_nn='cnn', mouth_features_dim=512, lstm_units_1=32, dense_fc_1=128, dense_fc_2=64,
                      conv_f_1=32, conv_f_2=64, conv_f_3=128):

    mouth_input_shape = (MOUTH_H, MOUTH_W, MOUTH_CHANNELS)
    my_input_mouth_images = Input(shape=(TIME_STEPS, *mouth_input_shape))
    my_input_head_poses = Input(shape=(TIME_STEPS, 3))
    my_input_n_of_frames = Input(shape=(1,))
    my_input_lipreader_dense = Input(shape=(1024,))
    my_input_lipreader_softmax = Input(shape=(500,))

    if mouth_nn == 'cnn':
        mouth_feature_model = my_timedistributed_cnn_model((TIME_STEPS, *mouth_input_shape), conv_f_1, conv_f_2, conv_f_3, mouth_features_dim)
    elif mouth_nn == 'resnet18':
        mouth_feature_model = ResnetBuilder.build_resnet_18(mouth_input_shape, mouth_features_dim)
    elif mouth_nn == 'resnet34':
        mouth_feature_model = ResnetBuilder.build_resnet_34(mouth_input_shape, mouth_features_dim)
    elif mouth_nn == 'resnet50':
        mouth_feature_model = ResnetBuilder.build_resnet_50(mouth_input_shape, mouth_features_dim)
    elif mouth_nn == 'resnet101':
        mouth_feature_model = ResnetBuilder.build_resnet_101(mouth_input_shape, mouth_features_dim)
    elif mouth_nn == 'resnet152':
        mouth_feature_model = ResnetBuilder.build_resnet_152(mouth_input_shape, mouth_features_dim)

    cnn_features = mouth_feature_model(my_input_mouth_images)

    lstm_input = Concatenate()([cnn_features, my_input_head_poses])

    lstm_output = LSTM(lstm_units_1, activation='tanh', kernel_regularizer=l2(1.e-4), return_sequences=False)(lstm_input)

    concatenated_features = Concatenate()([lstm_output, my_input_n_of_frames, my_input_lipreader_dense, my_input_lipreader_softmax])

    fc1 = Dense(dense_fc_1, activation='relu', kernel_regularizer=l2(1.e-4))(concatenated_features)

    fc2 = Dense(dense_fc_2, activation='relu', kernel_regularizer=l2(1.e-4))(fc1)

    # fc1 = Dense(dense_fc_1, activation='relu', kernel_regularizer=l2(1.e-4))(concatenated_features)

    # fc2 = Dense(dense_fc_2, activation='relu', kernel_regularizer=l2(1.e-4))(fc1)

    assessor_output = Dense(1, activation='sigmoid')(fc2)

    assessor = Model(inputs=[my_input_mouth_images, my_input_head_poses, my_input_n_of_frames, my_input_lipreader_dense, my_input_lipreader_softmax],
                     outputs=assessor_output)

    return assessor


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
        print(model_layer_index)
        if model_layer_index == 0:
            time_distributed_model.add(TimeDistributed(model.layers[model_layer_index], input_shape=(TIME_STEPS, *input_shape)))
        elif not isinstance(model.layers[model_layer_index], Add):
            time_distributed_model.add(TimeDistributed(model.layers[model_layer_index]))
        else:
            print("  Add layer!")
            time_distributed_model_add_layer_inputs = []
            # For each add layer input
            for add_layer_input in model.layers[model_layer_index].input:
                # Find the add layer input
                for l in range(model_layer_index):
                    # print(model.layers[l].output == add_layer_input, model.layers[l].output, add_layer_input)
                    if model.layers[l].output == add_layer_input:
                        time_distributed_model_add_layer_inputs.append(time_distributed_model.layers[l].output)
                        continue
            # Add them
            my_added = Add()(time_distributed_model_add_layer_inputs)
            time_distributed_model = Model(inputs=[time_distributed_model.input], outputs=[my_added])
    # Return
    return time_distributed_model


def make_time_distributed(model, TIME_STEPS, input_shape, verbose=True):
    extra_models = []
    extra_time_distributed_models = []
    model_input = model.input
    time_distributed_model_input = Input(shape=(TIME_STEPS, *input_shape))
    # For each layer
    for model_layer_index in range(1, len(model.layers)):
        if verbose:
            print(model_layer_index)
            print(model.layers[22].output)
        # If first layer, use time_distributed_model_input as input
        if model_layer_index == 1:
            x = TimeDistributed(model.layers[model_layer_index])(time_distributed_model_input)
            time_distributed_model = Model(inputs=[time_distributed_model_input], outputs=[x])
        # Else, in case it is not an Add layer,
        elif not isinstance(model.layers[model_layer_index], Add):
            # If the previous layer's output is this layer's input, make TimeDistributed
            if model.layers[model_layer_index-1].output == model.layers[model_layer_index].input:
                if verbose:
                    print("Prev layer's output is this layer's input")
                x = TimeDistributed(model.layers[model_layer_index])(time_distributed_model.layers[model_layer_index-1].output)
                time_distributed_model = Model(inputs=[time_distributed_model_input], outputs=[x])
            # Else, (prev layer's output is not this layer's input)
            else:
                if verbose:
                    print("This layer's input is not previous layer's output.")
                # Find the layer_index whose output is this layer's input, and use that
                # First search within model
                if verbose:
                    print("Searching in prev layers...")
                found = False
                for l in range(model_layer_index):
                    if model.layers[l].output == model.layers[model_layer_index].input:
                        if verbose:
                            print("Found at", l)
                        found = True
                        break
                if found == True:
                    this_model_layer_input = model.layers[l].output
                    this_time_distributed_layer_input = time_distributed_model.layers[l].output
                else:
                    if verbose:
                        print("Searching in extras...")
                    for extra_output_index in range(len(extra_models)):
                        if extra_models[extra_output_index].output == model.layers[model_layer_index].input:
                            if verbose:
                                print("Found at", extra_output_index)
                            this_model_layer_input = extra_models[extra_output_index].output
                            this_time_distributed_layer_input = extra_time_distributed_models[extra_output_index].output
                            break
                # Add this layer's outputs as extra output
                if verbose:
                    print("Appending to extra_outputs")
                y = model.layers[model_layer_index](this_model_layer_input)
                this_extra_model = Model(inputs=[model_input], outputs=[y])
                extra_models.append(this_extra_model)
                z = TimeDistributed(model.layers[model_layer_index])(this_time_distributed_layer_input)
                this_extra_time_distributed_model = Model(inputs=[time_distributed_model_input], outputs=[z])
                extra_time_distributed_models.append(this_extra_time_distributed_model)
                if verbose:
                    print("extra_models", extra_models)
                    print("extra_time_distributed_models", extra_time_distributed_models)
                # # # DONT DO THIS # # #
                # # Use the found layer input to append time_distributed_model
                # x = TimeDistributed(model.layers[model_layer_index])(this_layer_input)
                # time_distributed_model = Model(inputs=[time_distributed_model_input], outputs=[x])
                # # # DONT DO THIS # # #
        # ADD layer
        else:
            if verbose:
                print("  Add layer!")
            time_distributed_model_add_layer_inputs = []
            # First search within model
            found = False
            # For each add layer input
            for add_layer_input_index, add_layer_input in enumerate(model.layers[model_layer_index].input):
                # Find the add layer input
                if verbose:
                    print("Searching add layer input", add_layer_input_index, "in prev layers...")
                for l in range(model_layer_index):
                    # print(model.layers[l].output == add_layer_input, model.layers[l].output, add_layer_input)
                    if model.layers[l].output == add_layer_input:
                        if verbose:
                            print("Found at", l)
                        found = True
                        break
                if found == True:
                    time_distributed_model_add_layer_inputs.append(time_distributed_model.layers[l].output)
                # If not found
                else:
                    # Search in extras
                    print("Searching in extras...")
                    for extra_output_index in range(len(extra_models)):
                        if extra_models[extra_output_index].output == add_layer_input:
                            if verbose:
                                print("Found at", extra_output_index)
                            time_distributed_model_add_layer_inputs.append(extra_time_distributedmodels[extra_output_index].output)
                            break
            # Add them
            x = Add()(time_distributed_model_add_layer_inputs)
            time_distributed_model = Model(inputs=[time_distributed_model_input], outputs=[x])
    # Finally
    time_distributed_model = Model(inputs=[time_distributed_model_input], outputs=[x])
    # Return
    return time_distributed_model


# # Check which layer's input has this output
# for l in range(len(resnet_model.layers)):
#     if isinstance(resnet_model.layers[l].input, tf.Tensor):
#             if resnet_model.layers[22].output == resnet_model.layers[l].input:
#                 l
#                 break
#     else:
#             if resnet_model.layers[22].output in resnet_model.layers[l].input:
#                 print("multiple")
#                 l
#                 break


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

