import csv
import glob
import math
import numpy as np
import os
import tqdm

from assessor_params import *


def extract_and_save_word_set_nOfFramesPerWord(dataDir=LRW_DATA_DIR):
    # GET FRAME_DURATION OF WORDS
    frame_lengths = []
    # WORD
    for wordDir in tqdm.tqdm(sorted(glob.glob(os.path.join(dataDir, '*/')))):
        word = wordDir.split('/')[-2]
        # set
        for setDir in sorted(glob.glob(os.path.join(wordDir, '*/'))):
            setD = setDir.split('/')[-2]
            # number
            frame_lengths.append([])
            frame_lengths[-1].append(word)
            frame_lengths[-1].append(setD)
            for wordFileName in sorted(glob.glob(os.path.join(setDir, '*.txt'))):
                # wordFrameNumbers
                wordDuration = extract_word_duration(wordFileName)
                wordFrameDuration = math.ceil(VIDEO_FRAMES_PER_WORD/2 + wordDuration*VIDEO_FPS/2) - math.floor(VIDEO_FRAMES_PER_WORD/2 - wordDuration*VIDEO_FPS/2) + 1
                frame_lengths[-1].append(wordFrameDuration)
        save_list_of_lists_as_csv(frame_lengths, "frames_per_word")


def extract_word_duration(wordFileName):
    # Read last line of word metadata
    with open(wordFileName) as f:
        for line in f:
            pass
    # Find the duration of the word_metadata`
    return float(line.rstrip().split()[-2])


def save_list_of_lists_as_csv(list_of_lists, csv_file_name):
    with open(csv_file_name+".csv", "w") as f:
        wr = csv.writer(f)
        wr.writerows(list_of_lists)


def load_frames_per_word(csv_file_name):
    with open(csv_file_name, 'r') as f:  #opens PW file
        reader = csv.reader(f)
        data = list(list([i for i in rec]) for rec in csv.reader(f, delimiter=',')) #reads csv into a list of lists
        return data


def load_array_of_frames_per_word(csv_file_name):
    data = load_frames_per_word(csv_file_name)
    frames_per_word_test = []
    frames_per_word_train = []
    frames_per_word_val = []
    for d in data:
        # Test
        if d[1] == 'test':
            for i, e in enumerate(d):
                if i < 2:
                    continue
                frames_per_word_test.append(int(e))
        # Train
        if d[1] == 'train':
            for i, e in enumerate(d):
                if i < 2:
                    continue
                frames_per_word_train.append(int(e))
        # Val
        if d[1] == 'val':
            for i, e in enumerate(d):
                if i < 2:
                    continue
                frames_per_word_val.append(int(e))
    return np.array(frames_per_word_test), np.array(frames_per_word_train), np.array(frames_per_word_val)
