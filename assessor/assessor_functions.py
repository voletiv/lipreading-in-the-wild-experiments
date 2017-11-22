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


def generate_assessor_data_batches(batch_size=64, data_dir=LRW_DATA_DIR, collect_type="val",
                                   shuffle=True, equal_classes=False, use_CNN_LSTM=True,
                                   grayscale_images=False, random_crop=True, random_flip=False,
                                   verbose=False, skip_batches=0):

    if grayscale_images:
        MOUTH_CHANNELS = 1

    print("Loading LRW", collect_type, "init vars for generation...")

    if use_CNN_LSTM:
        # Read lrw_word_set_num_file_names, to read images
        lrw_word_set_num_txt_file_names = read_lrw_word_set_num_file_names(collect_type=collect_type, collect_by='sample')

        # Read start_frames_per_sample
        lrw_start_frames_per_sample = load_array_of_var_per_sample_from_csv(csv_file_name=START_FRAMES_PER_SAMPLE_CSV_FILE, collect_type=collect_type, collect_by='sample')

        # Read head_poses_per_sample
        lrw_head_poses_per_sample = read_head_poses(collect_type=collect_type, collect_by='sample')

    # Read n_of_frames_per_sample
    lrw_n_of_frames_per_sample = load_array_of_var_per_sample_from_csv(csv_file_name=N_OF_FRAMES_PER_SAMPLE_CSV_FILE, collect_type=collect_type, collect_by='sample')

    # Read dense, softmax, one_hot_y
    lrw_lipreader_dense, lrw_lipreader_softmax, lrw_correct_one_hot_y_arg = load_dense_softmax_y(collect_type=collect_type)

    # Lipreader correct (True) or wrong (False)
    lrw_lipreader_correct_or_wrong = np.argmax(lrw_lipreader_softmax, axis=1) == lrw_correct_one_hot_y_arg

    print("Loaded.")

    np.random.seed(29)
    full_index_list = np.arange(len(lrw_lipreader_correct_or_wrong))
    if use_CNN_LSTM:
        full_lrw_word_set_num_txt_file_names = list(lrw_word_set_num_txt_file_names)
        full_lrw_start_frames_per_sample = list(lrw_start_frames_per_sample)
        full_lrw_head_poses_per_sample = list(lrw_head_poses_per_sample)
    full_lrw_n_of_frames_per_sample = list(lrw_n_of_frames_per_sample)
    full_lrw_lipreader_dense = np.array(lrw_lipreader_dense)
    full_lrw_lipreader_softmax = np.array(lrw_lipreader_softmax)
    full_lrw_correct_one_hot_y_arg = np.array(lrw_correct_one_hot_y_arg)
    full_lrw_lipreader_correct_or_wrong = np.array(lrw_lipreader_correct_or_wrong)

    if equal_classes:
        lrw_lipreader_correct_idx = np.where(full_lrw_lipreader_correct_or_wrong == True)[0]
        lrw_lipreader_wrong_idx = np.where(full_lrw_lipreader_correct_or_wrong == False)[0]
        equal_classes_length = min(len(lrw_lipreader_correct_idx), len(lrw_lipreader_wrong_idx))

    # TO GENERATE
    while 1:

        # Shuffle
        if shuffle:
            np.random.shuffle(full_index_list)

        # Make index list such that Y = alternating True-False
        if equal_classes:
            lrw_lipreader_correct_or_wrong = full_lrw_lipreader_correct_or_wrong[full_index_list]
            lrw_lipreader_correct_idx = np.where(lrw_lipreader_correct_or_wrong == True)[0]
            lrw_lipreader_wrong_idx = np.where(lrw_lipreader_correct_or_wrong == False)[0]
            equal_classes_idx_list_within_full_idx_list = []
            for i in range(equal_classes_length):
                equal_classes_idx_list_within_full_idx_list.append(lrw_lipreader_correct_idx[i])
                equal_classes_idx_list_within_full_idx_list.append(lrw_lipreader_wrong_idx[i])
            idx_list_within_full_idx_list = np.array(equal_classes_idx_list_within_full_idx_list)
        else:
            idx_list_within_full_idx_list = full_index_list

        # Make stuff
        if use_CNN_LSTM:
            lrw_word_set_num_txt_file_names = [full_lrw_word_set_num_txt_file_names[i] for i in full_index_list[idx_list_within_full_idx_list]]
            lrw_start_frames_per_sample = [full_lrw_start_frames_per_sample[i] for i in full_index_list[idx_list_within_full_idx_list]]
            lrw_head_poses_per_sample = [full_lrw_head_poses_per_sample[i] for i in full_index_list[idx_list_within_full_idx_list]]
        lrw_n_of_frames_per_sample = [full_lrw_n_of_frames_per_sample[i] for i in full_index_list[idx_list_within_full_idx_list]]
        lrw_lipreader_dense = full_lrw_lipreader_dense[full_index_list[idx_list_within_full_idx_list]]
        lrw_lipreader_softmax = full_lrw_lipreader_softmax[full_index_list[idx_list_within_full_idx_list]]
        lrw_correct_one_hot_y_arg = full_lrw_correct_one_hot_y_arg[full_index_list[idx_list_within_full_idx_list]]
        lrw_lipreader_correct_or_wrong = full_lrw_lipreader_correct_or_wrong[full_index_list[idx_list_within_full_idx_list]]

        n_batches = len(lrw_lipreader_correct_or_wrong) // batch_size

        # For each batch
        for batch in range(n_batches):

            # Skip some if mentioned
            if batch < skip_batches:
                continue

            if verbose:
                print("Batch", batch+1, "of", n_batches)

            if use_CNN_LSTM:
                # Batch word_txt_files
                batch_lrw_word_set_num_txt_file_names = lrw_word_set_num_txt_file_names[batch*batch_size:(batch + 1)*batch_size]

                # Batch start frames per sample
                batch_start_frames_per_sample = lrw_start_frames_per_sample[batch*batch_size:(batch + 1)*batch_size]

                # Batch head poses
                batch_head_poses_per_sample = lrw_head_poses_per_sample[batch*batch_size:(batch + 1)*batch_size]

                # Batch mouth images (X)
                batch_mouth_images = np.zeros((batch_size, TIME_STEPS, MOUTH_H, MOUTH_W, MOUTH_CHANNELS))

                # Batch head poses for training (H)
                batch_head_poses_per_sample_for_training = np.zeros((batch_size, TIME_STEPS, 3))

            # Batch number of frames per sample (F)
            batch_n_of_frames_per_sample = lrw_n_of_frames_per_sample[batch*batch_size:(batch + 1)*batch_size]

            # Batch dense (D)
            batch_dense_per_sample = lrw_lipreader_dense[batch*batch_size:(batch + 1)*batch_size]

            # Batch softmax (S)
            batch_softmax_per_sample = lrw_lipreader_softmax[batch*batch_size:(batch + 1)*batch_size]

            # Batch lipreader one_hot_y
            batch_lipreader_one_hot_y_arg_per_sample = lrw_correct_one_hot_y_arg[batch*batch_size:(batch + 1)*batch_size]

            # # Batch lipreader_correct_or_wrong (Y)
            # batch_lipreader_correct_or_wrong = np.zeros((batch_size,))

            if use_CNN_LSTM:

                # GENERATE SET OF IMAGES PER WORD
                # For each WORD
                for sample_idx_within_batch, word_txt_file in enumerate(batch_lrw_word_set_num_txt_file_names):

                    if verbose:
                        print("Sample", sample_idx_within_batch+1, "of", batch_size)

                    # Word frame numbers
                    word_frame_numbers = range(batch_start_frames_per_sample[sample_idx_within_batch],
                                               batch_start_frames_per_sample[sample_idx_within_batch] + batch_n_of_frames_per_sample[sample_idx_within_batch])

                    # For each frame in mouth images
                    frame_0_start_index = -1
                    set_crop_offset = True
                    set_random_flip = True
                    for jpg_name in sorted(glob.glob('.'.join(word_txt_file.split('.')[:-1]) + '*mouth*.jpg')):

                        # Frame number
                        frame_number = int(jpg_name.split('/')[-1].split('.')[0].split('_')[-2])

                        # Frame in word
                        if frame_number in word_frame_numbers:

                            # Increment frame count, for saving at right index
                            frame_0_start_index += 1

                            if verbose:
                                print(jpg_name)

                            # Set image grayscale option
                            if grayscale_images:
                                cv_option = cv2.IMREAD_GRAYSCALE
                            else:
                                cv_option = cv2.IMREAD_COLOR

                            # Read image
                            mouth_image = robust_imread(jpg_name, cv_option)

                            if set_crop_offset:
                                # Images have been saved at 120x120. We need them at MOUTH_HxMOUTH_W
                                # To crop it
                                if random_crop:
                                    h_offset = np.random.randint(low=0, high=mouth_image.shape[0]-MOUTH_H)
                                    w_offset = np.random.randint(low=0, high=mouth_image.shape[1]-MOUTH_W)
                                else:
                                    h_offset = (mouth_image.shape[0] - MOUTH_H) // 2
                                    w_offset = (mouth_image.shape[1] - MOUTH_W) // 2
                                if verbose:
                                    print("Crop offsets (h, w):", h_offset, w_offset)
                                # Reset
                                set_crop_offset = False

                            # Crop image
                            mouth_image = mouth_image[h_offset:h_offset+MOUTH_H, w_offset:w_offset+MOUTH_W]

                            # Set whether to flip or not
                            if set_random_flip:
                                if random_flip and np.random.choice(2):
                                    flip = True
                                else:
                                    flip = False
                                set_random_flip = False

                            # Flip
                            if flip:
                                mouth_image = mouth_image[:, ::-1]

                            # Add this image in reverse order into X
                            # eg. If there are 7 frames: 0 0 0 0 0 0 0 7 6 5 4 3 2 1
                            batch_mouth_images[sample_idx_within_batch][-frame_0_start_index-1] = np.reshape(mouth_image, (MOUTH_H, MOUTH_W, MOUTH_CHANNELS))

                            # Add head pose
                            batch_head_poses_per_sample_for_training[sample_idx_within_batch][-frame_0_start_index-1] = batch_head_poses_per_sample[sample_idx_within_batch][frame_0_start_index]

            # Batch number of frames per sample (F)
            batch_n_of_frames_per_sample = np.reshape(np.array(batch_n_of_frames_per_sample)/float(MAX_FRAMES_PER_WORD), (len(batch_n_of_frames_per_sample), 1))

            # Correct_or_wrong
            batch_lipreader_correct_or_wrong = np.array(np.argmax(batch_softmax_per_sample, axis=1) == batch_lipreader_one_hot_y_arg_per_sample, dtype=float)

            # Yield X, H, F, D, S, Y
            if use_CNN_LSTM:
                yield [batch_mouth_images, batch_head_poses_per_sample_for_training, batch_n_of_frames_per_sample, batch_dense_per_sample, batch_softmax_per_sample], [batch_lipreader_correct_or_wrong]
            else:
                yield [batch_n_of_frames_per_sample, batch_dense_per_sample, batch_softmax_per_sample], [batch_lipreader_correct_or_wrong]


def robust_imread(jpg_name, cv_option=cv2.IMREAD_COLOR):
    try:
        image = cv2.imread(jpg_name, cv_option) / 255.
        return image
    except TypeError:
        return robust_imread(jpg_name, cv_option)


'''
for sample_idx_within_batch, word_txt_file in enumerate(batch_lrw_word_set_num_txt_file_names):
    word_frame_numbers = range(batch_start_frames_per_sample[sample_idx_within_batch],batch_start_frames_per_sample[sample_idx_within_batch] + batch_n_of_frames_per_sample[sample_idx_within_batch])
    frame_0_start_index = -1
    set_crop_offset = True
    for jpg_name in sorted(glob.glob('.'.join(word_txt_file.split('.')[:-1]) + '*mouth*.jpg')):
        frame_number = int(jpg_name.split('/')[-1].split('.')[0].split('_')[-2])
        if frame_number in word_frame_numbers:
            frame_0_start_index += 1
            mouth_image = robust_imread(jpg_name)
            if set_crop_offset:
                if random_crop:
                     h_offset = np.random.randint(low=0, high=mouth_image.shape[0]-MOUTH_H)
                     w_offset = np.random.randint(low=0, high=mouth_image.shape[1]-MOUTH_W)
                else:
                     h_offset = (mouth_image.shape[0] - MOUTH_H) / 2
                     w_offset = (mouth_image.shape[1] - MOUTH_W) / 2
                set_crop_offset = False
            mouth_image = mouth_image[h_offset:h_offset+MOUTH_H, w_offset:w_offset+MOUTH_W]
            batch_mouth_images[sample_idx_within_batch][-frame_0_start_index-1] = mouth_image
            batch_head_poses_per_sample_for_training[sample_idx_within_batch][-frame_0_start_index-1] = batch_head_poses_per_sample[sample_idx_within_batch][frame_0_start_index]
    break
batch_n_of_frames_per_sample = np.array(batch_n_of_frames_per_sample)/float(MAX_FRAMES_PER_WORD)
'''

#############################################################
# lrw_word_set_num_txt_file_names
#############################################################


def collect_all_lrw_word_set_num_txt_file_names(data_dir=LRW_DATA_DIR):

    # COLLECT NAMES OF ALL WORDS (so it can be shuffled) as .txt file names
    print("Collecting", collect_type, "lrw_word_names")

    lrw_word_set_num_txt_file_names = []
    lrw_word_set_num_txt_file_names_test = []
    lrw_word_set_num_txt_file_names_train = []
    lrw_word_set_num_txt_file_names_val = []

    # word
    for word_dir in tqdm.tqdm(sorted(glob.glob(os.path.join(data_dir, '*/')))):
        # set
        for set_dir in sorted(glob.glob(os.path.join(word_dir, '*/'))):
            # number
            for word_set_num_txt_file_name in sorted(glob.glob(os.path.join(set_dir, '*.txt'))):
                # Collect
                lrw_word_set_num_txt_file_names.append(word_set_num_txt_file_name)
                # Collect only test, train, or val
                if 'test' in word_set_num_txt_file_name:
                    lrw_word_set_num_txt_file_names_test.append(word_set_num_txt_file_name)
                if 'train' in word_set_num_txt_file_name:
                    lrw_word_set_num_txt_file_names_train.append(word_set_num_txt_file_name)
                if 'val' in word_set_num_txt_file_name:
                    lrw_word_set_num_txt_file_names_val.append(word_set_num_txt_file_name)

    write_list_as_txt_file("lrw_word_set_num_txt_file_names", lrw_word_set_num_txt_file_names)
    write_list_as_txt_file("lrw_word_set_num_txt_file_names_test", lrw_word_set_num_txt_file_names_test)
    write_list_as_txt_file("lrw_word_set_num_txt_file_names_train", lrw_word_set_num_txt_file_names_train)
    write_list_as_txt_file("lrw_word_set_num_txt_file_names_val", lrw_word_set_num_txt_file_names_val)


def read_lrw_word_set_num_file_names(collect_type="train", collect_by='sample'):
    if collect_by == 'sample':
        if 'test' in collect_type:
            return read_txt_file_as_list('lrw_word_set_num_txt_file_names_test.txt')
        elif 'train' in collect_type:
            return read_txt_file_as_list('lrw_word_set_num_txt_file_names_train.txt')
        elif 'val' in collect_type:
            return read_txt_file_as_list('lrw_word_set_num_txt_file_names_val.txt')
        else:
            return read_txt_file_as_list('lrw_word_set_num_txt_file_names.txt')
    elif collect_by == 'vocab_word':
        if 'test' in collect_type:
            return read_txt_file_as_list_per_vocab_word('lrw_word_set_num_txt_file_names_test.txt')
        elif 'train' in collect_type:
            return read_txt_file_as_list_per_vocab_word('lrw_word_set_num_txt_file_names_train.txt')
        elif 'val' in collect_type:
            return read_txt_file_as_list_per_vocab_word('lrw_word_set_num_txt_file_names_val.txt')
        else:
            return read_txt_file_as_list_per_vocab_word('lrw_word_set_num_txt_file_names.txt')


def write_list_as_txt_file(file_name, the_list):
    with open(file_name+'.txt', 'w') as f:
        for line in the_list:
            a = f.write(line + '\n')


def read_txt_file_as_list(file_name):
    the_list = []
    with open(file_name, 'r') as f:
        for line in f:
            the_list.append(line.rstrip())
    return the_list


def read_txt_file_as_list_per_vocab_word(file_name):
    the_list = []
    with open(file_name, 'r') as f:
        prev_word = ""
        for line in f:
            # New row for every vocab_word
            word = line.rstrip().split('/')[-1].split('.')[0].split('_')[0]
            if word != prev_word:
                the_list.append([])
                prev_word = word
            # Append
            the_list[-1].append(line.rstrip())
    return the_list


#############################################################
# DENSE, SOFTMAX, Y
#############################################################


def load_dense_softmax_y(collect_type):
    lrw_lipreader_dense_softmax_y = np.load(os.path.join(LRW_ASSESSOR_DIR, 'LRW_'+collect_type+'_dense_softmax_y.npz'))
    lrw_lipreader_dense = lrw_lipreader_dense_softmax_y['lrw_'+collect_type+'_dense']
    lrw_lipreader_softmax = lrw_lipreader_dense_softmax_y['lrw_'+collect_type+'_softmax']
    lrw_one_hot_y_rg = lrw_lipreader_dense_softmax_y['lrw_correct_one_hot_arg']
    return lrw_lipreader_dense, lrw_lipreader_softmax, lrw_one_hot_y_rg


#############################################################
# READ NUMBER OF FRAMES IN EVERY WORD
#############################################################


def load_array_of_var_per_sample_from_csv(csv_file_name=N_OF_FRAMES_PER_SAMPLE_CSV_FILE, collect_type=" ", collect_by="sample"):
    data = load_list_of_lists_of_frames_per_set(csv_file_name)
    frames_per = []
    prev_word = ""
    for d in data:
        if collect_by == 'vocab_word':
            # New row for every word
            word = d[0]
            if word != prev_word:
                frames_per.append([])
                prev_word = word
        # Test
        if 'test' in collect_type:
            if d[1] == 'test':
                for i, e in enumerate(d):
                    if i < 2:
                        continue
                    if collect_by == 'vocab_word':
                        frames_per[-1].append(int(e))
                    elif collect_by == 'sample':
                        frames_per.append(int(e))
        # Train
        elif 'train' in collect_type:
            if d[1] == 'train':
                for i, e in enumerate(d):
                    if i < 2:
                        continue
                    if collect_by == 'vocab_word':
                        frames_per[-1].append(int(e))
                    elif collect_by == 'sample':
                        frames_per.append(int(e))
        # Val
        elif 'val' in collect_type:
            if d[1] == 'val':
                for i, e in enumerate(d):
                    if i < 2:
                        continue
                    if collect_by == 'vocab_word':
                        frames_per[-1].append(int(e))
                    elif collect_by == 'sample':
                        frames_per.append(int(e))
        # All
        else:
            for i, e in enumerate(d):
                if i < 2:
                    continue
                if collect_by == 'vocab_word':
                    frames_per[-1].append(int(e))
                elif collect_by == 'sample':
                    frames_per.append(int(e))
    return frames_per


def load_list_of_lists_of_frames_per_set(csv_file_name):
    with open(csv_file_name, 'r') as f:  #opens PW file
        reader = csv.reader(f)
        data = list(list([i for i in rec]) for rec in csv.reader(f, delimiter=',')) #reads csv into a list of lists
        return data


# def extract_frame_idx_per_sample(lrw_word_set_num_txt_file_names):
#     frame_idx_per_sample = []
#     for word_set_num_txt_file in lrw_word_set_num_txt_file_names:
#         frame_idx_per_sample.append(extract_word_frame_numbers(word_set_num_txt_file, verbose=False))
#     return frame_idx_per_sample


#############################################################
# READ HEAD POSE
#############################################################


def read_head_poses(collect_type="val", collect_by="sample"):
    head_poses_per_frame = read_head_poses_per_frame(collect_type=collect_type)
    if collect_by == 'frame':
        return head_poses_per_frame
    elif collect_by == 'sample':
        lrw_frames_per_sample = load_array_of_var_per_sample_from_csv(csv_file_name=N_OF_FRAMES_PER_SAMPLE_CSV_FILE, collect_type=collect_type, collect_by='sample')
        head_poses = []
        def gen_head_pose(head_poses_per_frame):
            for i in range(len(head_poses_per_frame)):
                yield head_poses_per_frame[i]
        head_poses_gen = gen_head_pose(head_poses_per_frame)
        for i in range(len(lrw_frames_per_sample)):
            head_poses.append([])
            for n_of_frames in range(lrw_frames_per_sample[i]):
                try:
                    head_poses[-1].append(next(head_poses_gen))
                except StopIteration:
                    print("Iteration stopped! Something wrong, maybe?")
        return head_poses


def read_head_poses_per_frame(collect_type="val"):
    head_pose_files_list = read_head_pose_files_list(collect_type=collect_type)
    head_poses = np.empty((0, 3))
    for head_pose_file in head_pose_files_list:
        head_poses = np.vstack((head_poses, np.load(head_pose_file)))
    return head_poses


def read_head_pose_files_list(collect_type="val"):
    if 'test' in collect_type:
        return sorted(glob.glob(os.path.join(LRW_HEAD_POSE_DIR, "*_test.npy")))
    elif 'train' in collect_type:
        return sorted(glob.glob(os.path.join(LRW_HEAD_POSE_DIR, "*_train.npy")))
    elif 'val' in collect_type:
        return sorted(glob.glob(os.path.join(LRW_HEAD_POSE_DIR, "*_val.npy")))
    else:
        return [f for f in sorted(glob.glob(os.path.join(LRW_HEAD_POSE_DIR, "*.npy"))) if '_test.npy' not in f and '_train.npy' not in f and '_val.npy' not in f]


#############################################################
# COMPUTE NUMBER OF FRAMES IN EVERY WORD
#############################################################

# # EXTRACT AND SAVE N_OF_FRAMES IN EVERY WORD
# extract_and_save_word_set_nOfFramesPerWord(dataDir=LRW_DATA_DIR)


def extract_and_save_sample_start_frame_idx(dataDir=LRW_DATA_DIR):
    # GET FRAME_DURATION OF WORDS
    start_frame = []
    # WORD
    for word_dir in tqdm.tqdm(sorted(glob.glob(os.path.join(dataDir, '*/')))):
        word = word_dir.split('/')[-2]
        # set
        for set_dir in sorted(glob.glob(os.path.join(word_dir, '*/'))):
            setD = set_dir.split('/')[-2]
            # number
            start_frame.append([])
            start_frame[-1].append(word)
            start_frame[-1].append(setD)
            for word_set_num_txt_file_name in sorted(glob.glob(os.path.join(set_dir, '*.txt'))):
                # wordFrameNumbers
                word_start_frame_index = extract_word_frame_numbers(word_set_num_txt_file_name)[0]
                start_frame[-1].append(word_start_frame_index)
        save_list_of_lists_as_csv(start_frame, "start_frames_per_sample")


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
            for word_set_num_txt_file_name in sorted(glob.glob(os.path.join(set_dir, '*.txt'))):
                # wordFrameNumbers
                wordDuration = extract_word_duration(word_set_num_txt_file_name)
                wordFrameDuration = math.ceil(VIDEO_FRAMES_PER_WORD/2 + wordDuration*VIDEO_FPS/2) - math.floor(VIDEO_FRAMES_PER_WORD/2 - wordDuration*VIDEO_FPS/2) + 1
                frame_lengths[-1].append(wordFrameDuration)
        save_list_of_lists_as_csv(frame_lengths, "n_of_frames_per_sample")


def extract_word_frame_numbers(word_set_num_txt_file_name, verbose=False):
    # Find the duration of the word_metadata
    wordDuration = extract_word_duration(word_set_num_txt_file_name)
    # Find frame numbers
    wordFrameNumbers = range(math.floor(VIDEO_FRAMES_PER_WORD/2 - wordDuration*VIDEO_FPS/2),
        math.ceil(VIDEO_FRAMES_PER_WORD/2 + wordDuration*VIDEO_FPS/2) + 1)
    if verbose:
        print("Word frame numbers = ", wordFrameNumbers, "; Word duration = ", wordDuration)
    return wordFrameNumbers


def extract_word_duration(word_set_num_txt_file_name):
    # Read last line of word metadata
    with open(word_set_num_txt_file_name) as f:
        for line in f:
            pass
    # Find the duration of the word_metadata`
    return float(line.rstrip().split()[-2])


def save_list_of_lists_as_csv(list_of_lists, csv_file_name):
    with open(csv_file_name+".csv", "w") as f:
        wr = csv.writer(f)
        wr.writerows(list_of_lists)


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
            for word_set_num_txt_file_name in sorted(glob.glob(os.path.join(set_dir, '*.txt'))):
                n_words += len(extract_word_frame_numbers(word_set_num_txt_file_name, verbose=False))
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
        head_pose_val = head_pose_word[n_mouths_in_word_set[word_num][0]+n_mouths_in_word_set[word_num][1]:]
        # Save
        np.save(os.path.join(LRW_HEAD_POSE_DIR, 'head_pose_'+word+'_test.npy'), head_pose_test)
        np.save(os.path.join(LRW_HEAD_POSE_DIR, 'head_pose_'+word+'_train.npy'), head_pose_train)
        np.save(os.path.join(LRW_HEAD_POSE_DIR, 'head_pose_'+word+'_val.npy'), head_pose_val)
