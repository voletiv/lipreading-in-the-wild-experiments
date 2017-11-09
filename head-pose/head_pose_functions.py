from __future__ import print_function

import glob
import math
import os
import tqdm

from head_pose_params import *


# To write_frame_jpg_file_names_in_txt_file
def write_frame_jpg_file_names_in_txt_file(dataDir, startWord=None, startSetWordNumber=None, endWord=None, endSetWordNumber=None):
    # Start, end
    if startSetWordNumber is None and startWord is None:
        startExtracting = True
    else:
        startExtracting = False
    # MAKE .txt FILES OF ALL JPG FILES IN WORD
    f = open("head_pose_dummy.txt", 'w')
    # word
    for wordDir in sorted(glob.glob(os.path.join(dataDir, '*/'))):
            # print(wordDir, end='\r')
            # Start from that Word specified
            if startWord is not None:
                if startWord in wordDir:
                    startExtracting = True
                if not startExtracting:
                    continue
            # End at that Word specified
            if endWord is not None:
                if endWord in wordDir:
                    raise KeyboardInterrupt
            f, file_name = close_and_open_new_f(f, wordDir)
            # set: {test, train, val}
            for setDir in sorted(glob.glob(os.path.join(wordDir, '*/'))):
                # print(setDir, end='\r')
                for wordFileName in sorted(glob.glob(os.path.join(setDir, '*.txt'))):
                    # Extract word frame numbers
                    wordFrameNumbers = extract_word_frame_numbers(wordFileName)
                    # For all images
                    for jpgName in sorted(glob.glob('.'.join(wordFileName.split('.')[:-1]) + '*.jpg')):
                        # Start from that SetWordNumber specified
                        if startSetWordNumber is not None:
                            if startSetWordNumber in jpgName:
                                startExtracting = True
                        if not startExtracting:
                            continue
                        # End at that SetWordNumber specified
                        if endSetWordNumber is not None:
                            if endSetWordNumber in jpgName:
                                raise KeyboardInterrupt
                        # Consider only frame images, not mouth images
                        if "mouth.jpg" not in jpgName:
                            # Skip those frames that are not in the word
                            frameNumber = int(jpgName.split('/')[-1].split('.')[0].split('_')[-1])
                            if frameNumber not in wordFrameNumbers:
                                continue
                            a = f.write(jpgName + "\n")
                            print(jpgName, end="\r")
    f.close()


'''
for wordDir in sorted(glob.glob(os.path.join(LRW_DATA_DIR, '*/'))):
    for setDir in sorted(glob.glob(os.path.join(wordDir, '*/'))):
            for wordFileName in sorted(glob.glob(os.path.join(setDir, '*.txt'))):
                wordFrameNumbers = extract_word_frame_numbers(wordFileName, verbose=True)
                for jpgName in sorted(glob.glob('.'.join(wordFileName.split('.')[:-1]) + '*.jpg')):
                     if "mouth.jpg" not in jpgName:
                            frameNumber = int(jpgName.split('/')[-1].split('.')[0].split('_')[-1])
                            if frameNumber in wordFrameNumbers:
                                    print(jpgName)
                                    # print(jpgName, end="\r")
'''


# To make a new head_pose_jpg_file_names_<WORD>.txt file for each word
def close_and_open_new_f(f, wordDir):
    f.close()
    new_file_name = os.path.join(LRW_SAVE_DIR, "head_pose_jpg_file_names_{0}.txt".format(wordDir.split('/')[-2]))
    f = open(new_file_name, 'w')
    return f, new_file_name


def extract_word_frame_numbers(wordFileName, verbose=False):
    # Find the duration of the word_metadata
    wordDuration = extract_word_duration(wordFileName)
    # Find frame numbers
    wordFrameNumbers = range(math.floor(VIDEO_FRAMES_PER_WORD/2 - wordDuration*VIDEO_FPS/2),
        math.ceil(VIDEO_FRAMES_PER_WORD/2 + wordDuration*VIDEO_FPS/2) + 1)
    if verbose:
        print("Word frame numbers = ", wordFrameNumbers, "; Word duration = ", wordDuration)
    return wordFrameNumbers


def extract_word_duration(wordFileName):
    # Read last line of word metadata
    with open(wordFileName) as f:
        for line in f:
            pass
    # Find the duration of the word_metadata`
    return float(line.rstrip().split()[-2])


# To run_dlib_head_pose_estimator
def run_gazr_dlib_head_pose_estimator(dataDir):
    # Call shell command using subprocess
    # subprocess.Popen(["/home/voletiv/GitHubRepos/gazr/build/gazr_benchmark_head_pose_multiple_frames", "/home/voletiv/GitHubRepos/lipreading-in-the-wild-experiments/shape-predictor/shape_predictor_68_face_landmarks.dat", "head_pose.txt", ">", "a.txt"])
    # "> a.txt" doesn't work
    gazr_exe = os.path.join(GAZR_BUILD_DIR, "gazr_benchmark_head_pose_multiple_frames")
    for file in tqdm.tqdm(sorted(glob.glob(os.path.join(dataDir, 'head_pose_jpg_file_names*')))):
        head_pose_file_name = os.path.join(LRW_SAVE_DIR, "head_pose_{0}.txt".format(file.split('/')[-1].split('.')[0].split('_')[-1]))
        command = gazr_exe + " " + SHAPE_DAT_FILE + " " + file + " > " + head_pose_file_name
        os.system(command)

