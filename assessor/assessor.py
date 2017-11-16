import glob
import math
import os
import tqdm

from assessor_functions import *

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



