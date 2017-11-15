import csv
import cv2
import glob
import math
import numpy as np
import os
import tqdm

from assessor_params import *


#############################################################
# GENERATOR FUNCTION
#############################################################


def generate_lrw_mouth_image_batches(data_dir=LRW_DATA_DIR, batch_size=64, collect_type="train", shuffle=True, random_crop=True, verbose=False):

    # COLLECT NAMES OF ALL WORDS (so it can be shuffled) as .txt file names
    print("Collecting", collect_type, "lrw_word_names")

    lrw_word_txt_file_names = []

    # word
    for word_dir in tqdm.tqdm(sorted(glob.glob(os.path.join(data_dir, '*/')))):
        # set
        for set_dir in sorted(glob.glob(os.path.join(word_dir, '*/'))):
            # number
            for word_file_name in sorted(glob.glob(os.path.join(set_dir, '*.txt'))):
                # Collect only test, train, or val
                if ('test' in collect_type and 'test' not in word_file_name) or \
                        ('train' in collect_type and 'train' not in word_file_name) or \
                        ('val' in collect_type and 'val' not in word_file_name):
                    continue
                # Collect
                lrw_word_txt_file_names.append(word_file_name)

    lrw_word_txt_file_names = np.array(lrw_word_txt_file_names)

    # TO GENERATE
    while 1:

        # Shuffle
        if shuffle:
            np.random.shuffle(lrw_word_txt_file_names)

        n_batches = len(lrw_word_txt_file_names) // batch_size

        # For each batch
        for batch in range(n_batches):

            if verbose:
                print("Batch", batch+1, "of", n_batches)

            # Batch mouth images (X)
            batch_mouth_images = np.zeros((batch_size, TIME_STEPS, MOUTH_H, MOUTH_W, MOUTH_CHANNELS))

            # Batch one-hot words (Y)
            batch_one_hot_words = np.zeros((batch_size, LRW_VOCAB_SIZE))

            # word_txt_files of batch
            batch_lrw_word_txt_file_names = lrw_word_txt_file_names[batch*batch_size:(batch + 1)*batch_size]

            # GENERATE SET OF IMAGES PER WORD
            # For each WORD
            for word_idx_within_batch, word_txt_file in enumerate(batch_lrw_word_txt_file_names):

                if verbose:
                    print("Word", word_idx_within_batch+1, "of", batch_size)

                # Word frame numbers
                word_frame_numbers = extract_word_frame_numbers(word_txt_file, verbose=False)

                # For each frame in mouth images
                frame_count = 0
                set_crop_offset = True
                for jpg_name in sorted(glob.glob('.'.join(word_txt_file.split('.')[:-1]) + '*mouth*.jpg')):

                    # Frame number
                    frame_number = int(jpg_name.split('/')[-1].split('.')[0].split('_')[-2])

                    # Frame in word
                    if frame_number in word_frame_numbers:

                        # Increment frame count, for saving at right index
                        frame_count += 1

                        if verbose:
                            print(jpg_name)

                        # Read image
                        mouth_image = robust_imread(jpg_name)

                        if set_crop_offset:
                            # Images have been saved at 120x120. We need them at MOUTH_HxMOUTH_W
                            # To crop it
                            if random_crop:
                                h_offset = np.random.randint(low=0, high=mouth_image.shape[0]-MOUTH_H)
                                w_offset = np.random.randint(low=0, high=mouth_image.shape[1]-MOUTH_W)
                            else:
                                h_offset = (mouth_image.shape[0] - MOUTH_H) / 2
                                w_offset = (mouth_image.shape[1] - MOUTH_W) / 2
                            if verbose:
                                print("Crop offsets (h, w):", h_offset, w_offset)
                            # Reset
                            set_crop_offset = False

                        # Crop image
                        mouth_image = mouth_image[h_offset:h_offset+MOUTH_H, w_offset:w_offset+MOUTH_W]

                        # Add this image in reverse order into X
                        # eg. If there are 7 frames: 0 0 0 0 0 0 0 7 6 5 4 3 2 1
                        batch_mouth_images[word_idx_within_batch][-frame_count] = mouth_image

                        # MAKE ONE HOT WORDS
                        word_vocab_index = LRW_VOCAB.index(jpg_name.split('/')[-1].split('.')[0].split('_')[0])
                        batch_one_hot_words[word_idx_within_batch][word_vocab_index] = 1

            # Yield
            yield batch_mouth_images, batch_one_hot_words


def robust_imread(jpg_name, cv_option=cv2.IMREAD_COLOR):
    try:
        image = cv2.imread(jpg_name, cv_option) / 255.
        return image
    except TypeError:
        return robust_imread(jpg_name, cv_option)


#############################################################
# DENSE, SOFTMAX, Y
#############################################################


def load_dense_softmax_y(collect_type):
    LRW_dense_softmax_y = np.load(os.path.join(LRW_ASSESSOR_DIR, 'LRW_'+collect_type+'_dense_softmax_y.npz'))
    LRW_dense = LRW_dense_softmax_y[collect_type+'Dense']
    LRW_softmax = LRW_dense_softmax_y[collect_type+'Softmax']
    LRW_one_hot_y = LRW_dense_softmax_y[collect_type+'Y']
    return LRW_dense, LRW_softmax, LRW_one_hot_y


#############################################################
# NUMBER OF FRAMES IN EVERY WORD
#############################################################


def extract_and_save_word_set_nOfFramesPerWord(dataDir=LRW_DATA_DIR):
    # GET FRAME_DURATION OF WORDS
    frame_lengths = []
    # WORD
    for word_dir in tqdm.tqdm(sorted(glob.glob(os.path.join(dataDir, '*/')))):
        word = word_dir.split('/')[-2]
        # set
        for set_dir in sorted(glob.glob(os.path.join(word_dir, '*/'))):
            setD = set_dir.split('/')[-2]
            # number
            frame_lengths.append([])
            frame_lengths[-1].append(word)
            frame_lengths[-1].append(setD)
            for word_file_name in sorted(glob.glob(os.path.join(set_dir, '*.txt'))):
                # wordFrameNumbers
                wordDuration = extract_word_duration(word_file_name)
                wordFrameDuration = math.ceil(VIDEO_FRAMES_PER_WORD/2 + wordDuration*VIDEO_FPS/2) - math.floor(VIDEO_FRAMES_PER_WORD/2 - wordDuration*VIDEO_FPS/2) + 1
                frame_lengths[-1].append(wordFrameDuration)
        save_list_of_lists_as_csv(frame_lengths, "frames_per_word")


def extract_word_frame_numbers(word_file_name, verbose=False):
    # Find the duration of the word_metadata
    wordDuration = extract_word_duration(word_file_name)
    # Find frame numbers
    wordFrameNumbers = range(math.floor(VIDEO_FRAMES_PER_WORD/2 - wordDuration*VIDEO_FPS/2),
        math.ceil(VIDEO_FRAMES_PER_WORD/2 + wordDuration*VIDEO_FPS/2) + 1)
    if verbose:
        print("Word frame numbers = ", wordFrameNumbers, "; Word duration = ", wordDuration)
    return wordFrameNumbers


def extract_word_duration(word_file_name):
    # Read last line of word metadata
    with open(word_file_name) as f:
        for line in f:
            pass
    # Find the duration of the word_metadata`
    return float(line.rstrip().split()[-2])


def save_list_of_lists_as_csv(list_of_lists, csv_file_name):
    with open(csv_file_name+".csv", "w") as f:
        wr = csv.writer(f)
        wr.writerows(list_of_lists)


def load_array_of_frames_per_word(csv_file_name=N_FRAMES_PER_WORD_FILE):
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


def load_frames_per_word(csv_file_name):
    with open(csv_file_name, 'r') as f:  #opens PW file
        reader = csv.reader(f)
        data = list(list([i for i in rec]) for rec in csv.reader(f, delimiter=',')) #reads csv into a list of lists
        return data


#############################################################
# SPLIT HEAD POSE INTO TEST, TRAIN, VAL
#############################################################


def split_head_pose(data_dir=LRW_DATA_DIR):

    # COLLECT NUM OF WORDS IN EACH SET OF EACH WORD
    n_mouths_in_word_set = []
    # word
    for word_dir in tqdm.tqdm(sorted(glob.glob(os.path.join(data_dir, '*/')))):
        n_mouths_in_word_set.append([])
        # set
        for set_dir in sorted(glob.glob(os.path.join(word_dir, '*/'))):
            n_words = 0
            # number
            for word_file_name in sorted(glob.glob(os.path.join(set_dir, '*.txt'))):
                n_words += len(extract_word_frame_numbers(word_file_name, verbose=False))
            # Append n_words in set
            n_mouths_in_word_set[-1].append(n_words)

    # COLLECT ALL HEAD POSE FILE NAMES
    lrw_head_poses = []
    # Read head pose files
    for lrw_head_pose_file in tqdm.tqdm(sorted(glob.glob(os.path.join(LRW_HEAD_POSE_DIR, '*.npy')))):
        lrw_head_poses.append(np.load(lrw_head_pose_file))

    # SPLIT HEAD POSE INTO TEST, TRAIN, VAL
    for word_num in tqdm.tqdm(range(500)):
        # Collect word and head_pose of word
        word = LRW_VOCAB[word_num]
        head_pose_word = lrw_head_poses[word_num]
        # Split
        head_pose_test = head_pose_word[:n_mouths_in_word_set[word_num][0]]
        head_pose_train = head_pose_word[n_mouths_in_word_set[word_num][0]:n_mouths_in_word_set[word_num][0]+n_mouths_in_word_set[word_num][1]]
        head_pose_val = head_pose_word[n_mouths_in_word_set[word_num][0]+n_mouths_in_word_set[word_num][1]:n_mouths_in_word_set[word_num][0]+n_mouths_in_word_set[word_num][1]+n_mouths_in_word_set[word_num][2]]
        # Save
        np.save(os.path.join(LRW_HEAD_POSE_DIR, 'head_pose_'+word+'_test.npy'), head_pose_test)
        np.save(os.path.join(LRW_HEAD_POSE_DIR, 'head_pose_'+word+'_train.npy'), head_pose_train)
        np.save(os.path.join(LRW_HEAD_POSE_DIR, 'head_pose_'+word+'_val.npy'), head_pose_val)


