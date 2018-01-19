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
from syncnet_preds_functions import *

mouth_input_shape = (112, 112, 5)

# BATCH SIZE
batch_size = 50

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

# Found 20samplesPerWord_all500words
# Found next 10samplesPerWord_offset20_all500words
# Found next 10samplesPerWord_offset30_all500words
# Found next 10samplesPerWord_offset40_all500words
# Found next 10samplesPerWord_offset50_all500words

samples_per_word = 50

for offset in [250, 300, 350]:
    print("OFFSET", offset)
    # offset = 200
    # LRW_TRAIN, 200 per word
    train_lrw_word_set_num_txt_file_names = read_lrw_word_set_num_file_names(collect_type="train", collect_by='sample')
    n_batches = samples_per_word * 500 // batch_size
    train_generator = generate_syncnet_pred_data_batches(batch_size=batch_size, data_dir=data_dir, collect_type="train", shuffle=shuffle, equal_classes=equal_classes,
                                                         use_CNN_LSTM=use_CNN_LSTM, mouth_nn="syncnet", grayscale_images=grayscale_images, random_crop=random_crop, random_flip=random_flip, use_head_pose=use_head_pose,
                                                         use_softmax=use_softmax, use_softmax_ratios=use_softmax_ratios, verbose=verbose, skip_batches=0,
                                                         samples_per_word=samples_per_word, offset=offset)
    # Y
    train_Y = np.empty((0, TIME_STEPS, 128))
    # Predict
    for batch in tqdm.tqdm(range(n_batches)):
        ((X, _, _), _) = next(train_generator)
        y = syncnet_model.predict(X)
        train_Y = np.vstack((train_Y, y))
        if batch % 10 == 0:
            np.save('/shared/fusor/home/voleti.vikram/LRW_train_syncnet_preds_'+str(samples_per_word)+'samplesPerWord_'+str(offset)+'offset_all500words', train_Y)
    np.save('/shared/fusor/home/voleti.vikram/LRW_train_syncnet_preds_'+str(samples_per_word)+'samplesPerWord_'+str(offset)+'offset_all500words', train_Y)


def f(list_of_samples_per_word=[10]*((200-60)//10), list_of_offsets=range(60, 200, 10)):
    # LRW_TRAIN, 200 per word
train_lrw_word_set_num_txt_file_names = read_lrw_word_set_num_file_names(collect_type="train", collect_by='sample')
n_batches = samples_per_word * 500 // batch_size
for sample, offset in zip(list_of_samples_per_word, list_of_offsets):
    train_generator = generate_syncnet_pred_data_batches(batch_size=batch_size, data_dir=data_dir, collect_type="train", shuffle=shuffle, equal_classes=equal_classes,
                                                         use_CNN_LSTM=use_CNN_LSTM, mouth_nn="syncnet", grayscale_images=grayscale_images, random_crop=random_crop, random_flip=random_flip, use_head_pose=use_head_pose,
                                                         use_softmax=use_softmax, use_softmax_ratios=use_softmax_ratios, verbose=verbose, skip_batches=0,
                                                         samples_per_word=samples_per_word, offset=offset)
    # Y
    train_Y = np.empty((0, TIME_STEPS, 128))
    # Predict
    for batch in tqdm.tqdm(range(n_batches)):
        ((X, _, _), _) = next(train_generator)
        y = syncnet_model.predict(X)
        train_Y = np.vstack((train_Y, y))
        if batch % 10 == 0:
            print("Saving", '/shared/fusor/home/voleti.vikram/LRW_train_syncnet_preds_'+str(samples_per_word)+'samplesPerWord_'+str(offset)+'offset_all500words')
            np.save('/shared/fusor/home/voleti.vikram/LRW_train_syncnet_preds_'+str(samples_per_word)+'samplesPerWord_'+str(offset)+'offset_all500words', train_Y)
    np.save('/shared/fusor/home/voleti.vikram/LRW_train_syncnet_preds_'+str(samples_per_word)+'samplesPerWord_'+str(offset)+'offset_all500words', train_Y)




# LRW_VAL
train_lrw_word_set_num_txt_file_names = read_lrw_word_set_num_file_names(collect_type=train_collect_type, collect_by='sample')
n_batches = len(train_lrw_word_set_num_txt_file_names) // batch_size

train_generator = generate_syncnet_pred_data_batches(batch_size=batch_size, data_dir=data_dir, collect_type=train_collect_type, shuffle=shuffle, equal_classes=equal_classes,
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


# LRW_TEST
val_lrw_word_set_num_txt_file_names = read_lrw_word_set_num_file_names(collect_type=val_collect_type, collect_by='sample')
n_batches = len(val_lrw_word_set_num_txt_file_names) // batch_size

val_generator = generate_syncnet_pred_data_batches(batch_size=batch_size, data_dir=data_dir, collect_type=val_collect_type, shuffle=shuffle, equal_classes=equal_classes,
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
        # np.save('/shared/fusor/home/voleti.vikram/LRW_test_syncnet_preds_{0:03d}'.format(batch//10), val_Y)
        np.save('/shared/fusor/home/voleti.vikram/LRW_test_syncnet_preds', val_Y)
        # del val_Y
        # val_Y = np.empty((0, TIME_STEPS, 128))

np.save('/shared/fusor/home/voleti.vikram/LRW_test_syncnet_preds', val_Y)



# LRW_TRAIN
train_lrw_word_set_num_txt_file_names = read_lrw_word_set_num_file_names(collect_type="train", collect_by='sample')
n_batches = len(train_lrw_word_set_num_txt_file_names) // batch_size

skip_batches = 8711
lrw_train_generator = generate_syncnet_pred_data_batches(batch_size=batch_size, data_dir=data_dir, collect_type="train", shuffle=shuffle, equal_classes=equal_classes,
                                                   use_CNN_LSTM=use_CNN_LSTM, mouth_nn="syncnet", grayscale_images=grayscale_images, random_crop=random_crop, random_flip=random_flip, use_head_pose=use_head_pose,
                                                   use_softmax=use_softmax, use_softmax_ratios=use_softmax_ratios, verbose=verbose, skip_batches=skip_batches, get_last_smaller_batch=get_last_smaller_batch)

# Y
lrw_train_Y = np.empty((0, TIME_STEPS, 128))

# Predict
for batch in tqdm.tqdm(range(n_batches + 1)):
    if batch < skip_batches:
        continue
    ((X, _, _), _) = next(lrw_train_generator)
    y = syncnet_model.predict(X)
    lrw_train_Y = np.vstack((lrw_train_Y, y))
    if batch % 10 == 0:
        np.save('/shared/fusor/home/voleti.vikram/LRW_train_syncnet_preds_{0:03d}'.format(batch//10), lrw_train_Y)
        del lrw_train_Y
        lrw_train_Y = np.empty((0, TIME_STEPS, 128))

np.save('/shared/fusor/home/voleti.vikram/LRW_train_syncnet_preds_999', lrw_train_Y)



# SAVING THE FIRST 200 SAMPLES PER WORD
def get_first_n_samples_per_word_of_LRW_train_syncnet_preds(LRW_train_syncnet_preds_n_per_word, required_samples_per_word=200, verbose=False):
    train_lrw_word_set_num_txt_file_names = read_lrw_word_set_num_file_names(collect_type="train", collect_by='vocab_word')
    total_samples_per_word = []
    for word in range(500):
        total_samples_per_word.append(len(train_lrw_word_set_num_txt_file_names[word]))
    # Finding required sample idx
    # required_samples_per_word = 200
    required_samples_idx_per_word = []
    cumulative_num_of_samples = 0
    for w in range(500):
        required_samples_idx_per_word.append([])
        for i in range(required_samples_per_word):
            required_samples_idx_per_word[-1].append(cumulative_num_of_samples + i)
        samples_in_this_word = total_samples_per_word[w]
        cumulative_num_of_samples += samples_in_this_word
    # Finding idx in files
    LRW_train_syncnet_preds_files = sorted(glob.glob('/shared/fusor/home/voleti.vikram/LRW_train_syncnet_preds_[0-9][0-9][0-9].npy'))
    first_num_of_samples = np.load(LRW_train_syncnet_preds_files[0]).shape[0]
    common_num_of_samples = np.load(LRW_train_syncnet_preds_files[1]).shape[0]
    last_num_of_samples = np.load(LRW_train_syncnet_preds_files[-1]).shape[0]
    sample_idx_in_files = []
    cumulative_num_of_samples = 0
    for f in range(len(LRW_train_syncnet_preds_files)):
        sample_idx_in_files.append([])
        if f == 0:
            for i in range(first_num_of_samples):
                sample_idx_in_files[-1].append(cumulative_num_of_samples + i)
            cumulative_num_of_samples += first_num_of_samples
        elif f == len(LRW_train_syncnet_preds_files) - 1:
            for i in range(last_num_of_samples):
                sample_idx_in_files[-1].append(cumulative_num_of_samples + i)
            cumulative_num_of_samples += last_num_of_samples
        else:
            for i in range(common_num_of_samples):
                sample_idx_in_files[-1].append(cumulative_num_of_samples + i)
            cumulative_num_of_samples += common_num_of_samples
    # 200 per word
    def find_file_index(sample_index, file_index=-1):
        file_index += 1
        if sample_index in sample_idx_in_files[file_index]:
            return file_index
        else:
            return find_file_index(sample_index, file_index)
    verbose = False
    cumulative_num_of_samples = 0
    start_file_index = -1
    prev_end_file_index = -1
    # LRW_train_syncnet_preds_200_per_word = np.empty((0, 21, 128))
    for w in tqdm.tqdm(range(500)):
        if verbose:
            print("\n\nWORD", w+1)
        required_samples_idx = required_samples_idx_per_word[w]
        # print("required_samples_idx", required_samples_idx)
        required_samples_start_idx = required_samples_idx[0]
        # print("required_samples_start_idx", required_samples_start_idx)
        required_samples_end_idx = required_samples_idx[-1]
        # print("required_samples_end_idx", required_samples_end_idx)
        if verbose:
            print("Require samples from", required_samples_start_idx, "to", required_samples_end_idx)
        start_file_index = find_file_index(required_samples_start_idx, start_file_index)
        if verbose:
            print("start_file_index", start_file_index)
        if start_file_index == prev_end_file_index:
            read_start_file = False
        else:
            read_start_file = True
        if read_start_file:
            if verbose:
                print("Reading", LRW_train_syncnet_preds_files[start_file_index], "...")
            LRW_train_syncnet_preds_current = np.load(LRW_train_syncnet_preds_files[start_file_index])
        if verbose:
            print("File has indices from", sample_idx_in_files[start_file_index][0], "to", sample_idx_in_files[start_file_index][-1])
        start_sample_index = sample_idx_in_files[start_file_index].index(required_samples_start_idx)
        if verbose:
            print("start_sample_index", start_sample_index)
        if verbose:
            print("vstacking from", start_sample_index, "to", required_samples_per_word, "samples if possible")
        LRW_train_syncnet_preds_n_per_word = np.vstack((LRW_train_syncnet_preds_n_per_word, LRW_train_syncnet_preds_current[start_sample_index:start_sample_index+required_samples_per_word]))
        if required_samples_end_idx not in sample_idx_in_files[start_file_index]:
            if verbose:
                print("required_samples_end_idx", required_samples_end_idx, "not in the file. Appending end_file_index...")
            end_file_index = start_file_index + 1
            if verbose:
                print("end_file_index", end_file_index)
            read_end_file = True
        else:
            read_end_file = False
        if read_end_file:
            if verbose:
                print("Reading end file", LRW_train_syncnet_preds_files[end_file_index], "...")
            LRW_train_syncnet_preds_current = np.load(LRW_train_syncnet_preds_files[end_file_index])
            if verbose:
                print("File has indices from", sample_idx_in_files[end_file_index][0], "to", sample_idx_in_files[end_file_index][-1])
            end_sample_index = sample_idx_in_files[end_file_index].index(required_samples_end_idx)
            if verbose:
                print("Reading up to end_sample_index", end_sample_index)
            LRW_train_syncnet_preds_n_per_word = np.vstack((LRW_train_syncnet_preds_n_per_word, LRW_train_syncnet_preds_current[:end_sample_index+1]))
            prev_end_file_index = end_file_index
        else:
            prev_end_file_index = start_file_index
        if verbose:
            print(LRW_train_syncnet_preds_n_per_word.shape)

LRW_train_syncnet_preds_200_per_word = np.empty((0, 21, 128))
get_first_n_samples_per_word_of_LRW_train_syncnet_preds(LRW_train_syncnet_preds_200_per_word, required_samples_per_word=200, verbose=False):

np.save('LRW_train_syncnet_preds_200_per_word', LRW_train_syncnet_preds_200_per_word)
