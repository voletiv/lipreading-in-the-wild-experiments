import glob
import math
import os
import tqdm

from assessor_functions import *

######################################################
# NUMBER OF FRAMES IN EVERY WORD
######################################################

# # EXTRACT AND SAVE N_OF_FRAMES IN EVERY WORD
# extract_and_save_word_set_nOfFramesPerWord(dataDir=LRW_DATA_DIR)

# LOAD N_OF_FRAMES IN EVERY WORD
frames_every_word_test, frames_every_word_train, frames_every_word_val = load_array_of_frames_per_word(os.path.join(LRW_ASSESSOR_DIR, ))

######################################################
# DENSE, SOFTMAX, Y
######################################################

lrw_test_dense, LRW_test_softmax, LRW_test_one_hot_y = load_dense_softmax_y("test")
lrw_train_dense, LRW_train_softmax, LRW_train_one_hot_y = load_dense_softmax_y("train")
lrw_val_dense, LRW_val_softmax, LRW_val_one_hot_y = load_dense_softmax_y("val")

######################################################
# HEAD POSE
######################################################





######################################################
# GEN BATCHES OF IMAGES
######################################################

train_X_Y_gen = generate_lrw_mouth_image_batches(data_dir=LRW_DATA_DIR, batch_size=64, collect_type="train", shuffle=True, random_crop=True, verbose=False)


######################################################
# GEN BATCHES OF IMAGES
######################################################



