import csv
import cv2
import glob
import math
import numpy as np
import os
import tqdm

from assessor_functions import *


def generate_syncnet_pred_data_batches(batch_size=64, data_dir=".", collect_type="val", shuffle=True, equal_classes=False,
                                       use_CNN_LSTM=True, mouth_nn='cnn', grayscale_images=False, random_crop=True, random_flip=False, use_head_pose=True,
                                       use_softmax=True, use_softmax_ratios=True, verbose=False, skip_batches=0, get_last_smaller_batch=False,
                                       samples_per_word=None, offset=0):

    if grayscale_images:
        MOUTH_CHANNELS = 1

    if mouth_nn == 'syncnet':
        MOUTH_CHANNELS = 5

    print("Loading LRW", collect_type, "init vars for generation...")

    if samples_per_word is not None:

        # Sort idx by word
        lrw_word_set_num_txt_file_names = read_lrw_word_set_num_file_names(collect_type=collect_type, collect_by='sample')
        idx_by_word = []
        prev_name = ""
        for i, filename in enumerate(tqdm.tqdm(lrw_word_set_num_txt_file_names)):
            if os.path.normpath(filename).split('/')[-3] != prev_name:
                prev_name = os.path.normpath(filename).split('/')[-3]
                idx_by_word.append([])
            idx_by_word[-1].append(i)

        # Make sample_idx
        samples_idx = []
        for w in range(500):
            for i in range(samples_per_word):
                samples_idx.append(idx_by_word[w][offset + i])

    if use_CNN_LSTM:
        # Read lrw_word_set_num_file_names, to read images
        lrw_word_set_num_txt_file_names = read_lrw_word_set_num_file_names(collect_type=collect_type, collect_by='sample')
        if samples_per_word is not None:
            lrw_word_set_num_txt_file_names = [lrw_word_set_num_txt_file_names[i] for i in samples_idx]

        # Read start_frames_per_sample
        lrw_start_frames_per_sample = load_array_of_var_per_sample_from_csv(csv_file_name=START_FRAMES_PER_SAMPLE_CSV_FILE, collect_type=collect_type, collect_by='sample')
        if samples_per_word is not None:
            lrw_start_frames_per_sample = [lrw_start_frames_per_sample[i] for i in samples_idx]

        if mouth_nn == 'syncnet_preds':
            lrw_syncnet_preds = load_syncnet_preds(collect_type=collect_type)
            if samples_per_word is not None:
                lrw_syncnet_preds = lrw_syncnet_preds[samples_idx]

        if use_head_pose:
            # Read head_poses_per_sample
            lrw_head_poses_per_sample = read_head_poses(collect_type=collect_type, collect_by='sample')
            if samples_per_word is not None:
                lrw_head_poses_per_sample = lrw_head_poses_per_sample[samples_idx]

    # Read n_of_frames_per_sample
    lrw_n_of_frames_per_sample = load_array_of_var_per_sample_from_csv(csv_file_name=N_OF_FRAMES_PER_SAMPLE_CSV_FILE, collect_type=collect_type, collect_by='sample')
    if samples_per_word is not None:
        lrw_n_of_frames_per_sample = [lrw_n_of_frames_per_sample[i] for i in samples_idx]

    # Read dense, softmax, one_hot_y
    if collect_type != 'train':
        if verbose:
            print("Loading LRW", collect_type, "dense, softmax, y")
        lrw_lipreader_dense, lrw_lipreader_softmax, lrw_correct_one_hot_y_arg = load_dense_softmax_y(collect_type=collect_type)
    else:
        lrw_lipreader_dense = np.zeros((488766, 500))
        lrw_lipreader_softmax = np.zeros((488766, 500))
        lrw_correct_one_hot_y_arg = np.zeros((488766))

    if samples_per_word is not None:
        lrw_lipreader_dense = lrw_lipreader_dense[samples_idx]
        lrw_lipreader_softmax = lrw_lipreader_softmax[samples_idx]
        lrw_correct_one_hot_y_arg = lrw_correct_one_hot_y_arg[samples_idx]

    # Lipreader correct (True) or wrong (False)
    lrw_lipreader_correct_or_wrong = np.argmax(lrw_lipreader_softmax, axis=1) == lrw_correct_one_hot_y_arg

    if use_softmax_ratios:
        lrw_lipreader_softmax_ratios = load_softmax_ratios(collect_type=collect_type)
        if samples_per_word is not None:
            lrw_lipreader_softmax_ratios = lrw_lipreader_softmax_ratios[samples_idx]

    print("Loaded.")

    np.random.seed(29)
    full_index_list = np.arange(len(lrw_lipreader_correct_or_wrong))
    if use_CNN_LSTM:
        full_lrw_word_set_num_txt_file_names = list(lrw_word_set_num_txt_file_names)
        full_lrw_start_frames_per_sample = list(lrw_start_frames_per_sample)
        if mouth_nn == 'syncnet_preds':
            full_lrw_syncnet_preds = np.array(lrw_syncnet_preds)
        if use_head_pose:
            full_lrw_head_poses_per_sample = list(lrw_head_poses_per_sample)
    full_lrw_n_of_frames_per_sample = list(lrw_n_of_frames_per_sample)
    full_lrw_lipreader_dense = np.array(lrw_lipreader_dense)
    full_lrw_lipreader_softmax = np.array(lrw_lipreader_softmax)
    if use_softmax_ratios:
        full_lrw_lipreader_softmax_ratios = np.array(lrw_lipreader_softmax_ratios)
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
            if mouth_nn == 'syncnet_preds':
                lrw_syncnet_preds = full_lrw_syncnet_preds[full_index_list[idx_list_within_full_idx_list]]
            if use_head_pose:
                lrw_head_poses_per_sample = [full_lrw_head_poses_per_sample[i] for i in full_index_list[idx_list_within_full_idx_list]]
        lrw_n_of_frames_per_sample = [full_lrw_n_of_frames_per_sample[i] for i in full_index_list[idx_list_within_full_idx_list]]
        lrw_lipreader_dense = full_lrw_lipreader_dense[full_index_list[idx_list_within_full_idx_list]]
        lrw_lipreader_softmax = full_lrw_lipreader_softmax[full_index_list[idx_list_within_full_idx_list]]
        if use_softmax_ratios:
            lrw_lipreader_softmax_ratios = full_lrw_lipreader_softmax_ratios[full_index_list[idx_list_within_full_idx_list]]
        lrw_correct_one_hot_y_arg = full_lrw_correct_one_hot_y_arg[full_index_list[idx_list_within_full_idx_list]]
        lrw_lipreader_correct_or_wrong = full_lrw_lipreader_correct_or_wrong[full_index_list[idx_list_within_full_idx_list]]

        n_batches = len(lrw_word_set_num_txt_file_names) // batch_size

        if get_last_smaller_batch:
            if len(lrw_word_set_num_txt_file_names) > n_batches * batch_size:
                n_batches += 1

        if verbose:
            print("Total =", len(lrw_lipreader_correct_or_wrong), "; batch_size =", batch_size, "; n_batches =", n_batches, "; skip", skip_batches, "batches")

        # For each batch
        for batch in range(n_batches):

            if verbose:
                print("Batch", batch+1, "of", n_batches)

            # Skip some if mentioned
            if batch < skip_batches:
                if verbose:
                    print("Skipping it.")
                continue

            if use_CNN_LSTM:
                # Batch word_txt_files
                batch_lrw_word_set_num_txt_file_names = lrw_word_set_num_txt_file_names[batch*batch_size:(batch + 1)*batch_size]

                if mouth_nn == 'syncnet_preds':
                    batch_syncnet_preds = np.zeros((batch_size, TIME_STEPS, lrw_syncnet_preds.shape[-1]))

                else:
                    # Batch start frames per sample
                    batch_start_frames_per_sample = lrw_start_frames_per_sample[batch*batch_size:(batch + 1)*batch_size]

                    # Batch mouth images (X)
                    batch_mouth_images = np.zeros((batch_size, TIME_STEPS, MOUTH_H, MOUTH_W, MOUTH_CHANNELS))

                if use_head_pose:

                    # Batch head poses
                    batch_head_poses_per_sample = lrw_head_poses_per_sample[batch*batch_size:(batch + 1)*batch_size]

                    # Batch head poses for training (H)
                    batch_head_poses_per_sample_for_training = np.zeros((batch_size, TIME_STEPS, 3))

            # Batch number of frames per sample (F)
            batch_n_of_frames_per_sample = lrw_n_of_frames_per_sample[batch*batch_size:(batch + 1)*batch_size]

            # Batch dense (D)
            batch_dense_per_sample = lrw_lipreader_dense[batch*batch_size:(batch + 1)*batch_size]

            # Batch softmax (S)
            batch_softmax_per_sample = lrw_lipreader_softmax[batch*batch_size:(batch + 1)*batch_size]

            if use_softmax_ratios:
                # Batch softmax_ratios (R)
                batch_softmax_ratios_per_sample = lrw_lipreader_softmax_ratios[batch*batch_size:(batch + 1)*batch_size]

            # Batch lipreader one_hot_y
            batch_lipreader_one_hot_y_arg_per_sample = lrw_correct_one_hot_y_arg[batch*batch_size:(batch + 1)*batch_size]

            # # Batch lipreader_correct_or_wrong (Y)
            # batch_lipreader_correct_or_wrong = np.zeros((batch_size,))

            if use_CNN_LSTM:

                # HEAD_POSE
                if use_head_pose:
                    for sample_idx_within_batch in range(batch_size):
                        for time_step in range(len(batch_head_poses_per_sample[sample_idx_within_batch])):
                            batch_head_poses_per_sample_for_training[sample_idx_within_batch][-time_step-1] = batch_head_poses_per_sample[sample_idx_within_batch][time_step]

                if mouth_nn == 'syncnet_preds':
                    batch_syncnet_preds = lrw_syncnet_preds[batch*batch_size:(batch + 1)*batch_size]
                    batch_mouth_images = batch_syncnet_preds    # Just for continuity in naming

                else:

                    # GENERATE SET OF IMAGES PER WORD
                    # For each WORD
                    for sample_idx_within_batch, word_txt_file in enumerate(batch_lrw_word_set_num_txt_file_names):

                        if verbose:
                            print("Sample", sample_idx_within_batch+1, "of", batch_size)

                        # Word frame numbers
                        if mouth_nn == 'syncnet':
                            word_frame_numbers = list(range(batch_start_frames_per_sample[sample_idx_within_batch] - 2,
                                                            batch_start_frames_per_sample[sample_idx_within_batch] + batch_n_of_frames_per_sample[sample_idx_within_batch] + 2))
                        else:
                            word_frame_numbers = list(range(batch_start_frames_per_sample[sample_idx_within_batch],
                                                            batch_start_frames_per_sample[sample_idx_within_batch] + batch_n_of_frames_per_sample[sample_idx_within_batch]))

                        if verbose:
                            print("Frame numbers:", word_frame_numbers)
                            if mouth_nn == 'syncnet':
                                print("EXCLUDING first 2 and last 2")

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

                                # Don't increment frame count in case it's the 2 extra frames before and after actual frames of word
                                if mouth_nn == 'syncnet':
                                    if frame_number in [word_frame_numbers[0], word_frame_numbers[1], word_frame_numbers[-1], word_frame_numbers[-2]]:
                                        frame_0_start_index -= 1

                                if verbose:
                                    print(jpg_name)

                                # Set image grayscale option
                                if grayscale_images:
                                    cv_option = cv2.IMREAD_GRAYSCALE
                                else:
                                    cv_option = cv2.IMREAD_COLOR

                                # Read image
                                mouth_image = robust_imread(jpg_name, cv_option)

                                if mouth_nn != 'syncnet':
                                    mouth_image /= 255.

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

                                if mouth_nn == 'syncnet':
                                    # Add this image in reverse order into X
                                    # eg. If there are 7 frames: 0 0 0 0 0 0 0 7 6 5 4 3 2 1
                                    if frame_number == word_frame_numbers[0]:
                                        batch_mouth_images[sample_idx_within_batch][-1][:, :, 0] = np.reshape(mouth_image, (MOUTH_H, MOUTH_W))
                                    elif frame_number == word_frame_numbers[1]:
                                        batch_mouth_images[sample_idx_within_batch][-1][:, :, 1] = np.reshape(mouth_image, (MOUTH_H, MOUTH_W))
                                        batch_mouth_images[sample_idx_within_batch][-2][:, :, 0] = np.reshape(mouth_image, (MOUTH_H, MOUTH_W))
                                    elif frame_number == word_frame_numbers[-1]:
                                        batch_mouth_images[sample_idx_within_batch][-batch_n_of_frames_per_sample[sample_idx_within_batch]][:, :, 4] = np.reshape(mouth_image, (MOUTH_H, MOUTH_W))
                                    elif frame_number == word_frame_numbers[-2]:
                                        batch_mouth_images[sample_idx_within_batch][-batch_n_of_frames_per_sample[sample_idx_within_batch]][:, :, 3] = np.reshape(mouth_image, (MOUTH_H, MOUTH_W))
                                        batch_mouth_images[sample_idx_within_batch][-batch_n_of_frames_per_sample[sample_idx_within_batch]+1][:, :, 4] = np.reshape(mouth_image, (MOUTH_H, MOUTH_W))
                                    else:
                                        batch_mouth_images[sample_idx_within_batch][-frame_0_start_index-1][:, :, 2] = np.reshape(mouth_image, (MOUTH_H, MOUTH_W))
                                        if frame_0_start_index - 1 >= 0:
                                            batch_mouth_images[sample_idx_within_batch][(-frame_0_start_index-1) + 1][:, :, 3] = np.reshape(mouth_image, (MOUTH_H, MOUTH_W))
                                        if frame_0_start_index - 2 >= 0:
                                            batch_mouth_images[sample_idx_within_batch][(-frame_0_start_index-1) + 2][:, :, 4] = np.reshape(mouth_image, (MOUTH_H, MOUTH_W))
                                        if frame_0_start_index + 1 < TIME_STEPS:
                                            batch_mouth_images[sample_idx_within_batch][(-frame_0_start_index-1) - 1][:, :, 1] = np.reshape(mouth_image, (MOUTH_H, MOUTH_W))
                                        if frame_0_start_index + 2 < TIME_STEPS:
                                            batch_mouth_images[sample_idx_within_batch][(-frame_0_start_index-1) - 2][:, :, 0] = np.reshape(mouth_image, (MOUTH_H, MOUTH_W))

                                else:
                                    # Add this image in reverse order into X
                                    # eg. If there are 7 frames: 0 0 0 0 0 0 0 7 6 5 4 3 2 1
                                    batch_mouth_images[sample_idx_within_batch][-frame_0_start_index-1] = np.reshape(mouth_image, (MOUTH_H, MOUTH_W, MOUTH_CHANNELS))

            # Batch number of frames per sample (F)
            batch_n_of_frames_per_sample = np.reshape(np.array(batch_n_of_frames_per_sample)/float(MAX_FRAMES_PER_WORD), (len(batch_n_of_frames_per_sample), 1))

            # Correct_or_wrong
            batch_lipreader_correct_or_wrong = np.array(np.argmax(batch_softmax_per_sample, axis=1) == batch_lipreader_one_hot_y_arg_per_sample, dtype=float)

            if get_last_smaller_batch:
                if batch+1 == n_batches:
                    curr_batch_size = len(lrw_word_set_num_txt_file_names) - batch*batch_size
                    if use_CNN_LSTM:
                        batch_mouth_images = batch_mouth_images[:curr_batch_size]
                        if use_head_pose:
                            batch_head_poses_per_sample_for_training = batch_head_poses_per_sample_for_training[curr_batch_size]
                    batch_n_of_frames_per_sample = batch_n_of_frames_per_sample[:curr_batch_size]
                    batch_dense_per_sample = batch_dense_per_sample[:curr_batch_size]
                    if use_softmax:
                        batch_softmax_per_sample = batch_softmax_per_sample[:curr_batch_size]
                    if use_softmax_ratios:
                        batch_softmax_ratios_per_sample = batch_softmax_ratios_per_sample[:curr_batch_size]
                    batch_lipreader_correct_or_wrong = batch_lipreader_correct_or_wrong[:curr_batch_size]

            # Yield (X, H, F, D, S, R), Y
            X = []
            if use_CNN_LSTM:
                X += [batch_mouth_images]
                if use_head_pose:
                    X += [batch_head_poses_per_sample_for_training]
            X += [batch_n_of_frames_per_sample, batch_dense_per_sample]
            if use_softmax:
                X += [batch_softmax_per_sample]
            if use_softmax_ratios:
                X += [batch_softmax_ratios_per_sample]
            y = batch_lipreader_correct_or_wrong
            yield (X, y)


def robust_imread(jpg_name, cv_option=cv2.IMREAD_COLOR):
    try:
        image = cv2.imread(jpg_name, cv_option)
        return image
    except TypeError:
        return robust_imread(jpg_name, cv_option)

