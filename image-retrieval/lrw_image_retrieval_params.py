# PARAMS

import os
import sys

IR_DIR = os.path.dirname(os.path.realpath(__file__))

LRW_DIR = os.path.normpath(os.path.join('..', IR_DIR))

if LRW_DIR not in sys.path:
    sys.path.append(LRW_DIR)

for file in os.listdir(IR_DIR):
    if '.mat' in file:  # newest_retrieval_LRW_500.mat
        LRW_LIPREADER_OUTPUTS_MAT_FILE = file
    # if 'magnetar_LRW_all_words' in file:
    #     LRW_VOCAB_FILE = file
    if 'blazar_LRW_all_words' in file:
        LRW_VOCAB_FILE = file

LRW_VOCAB_SIZE = 500
LRW_TEST_SAMPLES_PER_CLASS = 50
LRW_TRAIN_SAMPLES_PER_CLASS = 1000
LRW_VAL_SAMPLES_PER_CLASS = 50
