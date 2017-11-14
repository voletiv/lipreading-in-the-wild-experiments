import glob
import math
import os

from assessor_functions import *

######################################################
# NUMBER OF FRAMES IN EVERY WORD
######################################################

# # EXTRACT AND SAVE N_OF_FRAMES IN EVERY WORD
# extract_and_save_word_set_nOfFramesPerWord(dataDir=LRW_DATA_DIR)

# LOAD N_OF_FRAMES IN EVERY WORD
frames_every_word_test, frames_every_word_train, frames_every_word_val = load_array_of_frames_per_word('/home/voletiv/GitHubRepos/frames_per_word.csv')

######################################################
# DENSE, SOFTMAX
######################################################



######################################################
# GEN BATCHES OF IMAGES
######################################################

list_of_file_names_per_image = []

# COLLECT SET OF IMAGES FOR EVERY WORD
# For each WORD
for word_txt_file in sorted(glob.glob(os.path.join(LRW_SAVE_DIR, '*.txt'))):
    list_of_file_names_per_image.append([])
    # Read all image names in word
    with open(word_txt_file) as f:
        lines = f.readlines()
    # Append all image names per setWordNumber
    list_of_file_names_per_image
        for line in f:














# WORD
for wordDir in sorted(glob.glob(os.path.join(LRW_DATA_DIR, '*/'))):
    # set
    for setDir in sorted(glob.glob(os.path.join(wordDir, '*/'))):
        # number
        for wordFileName in sorted(glob.glob(os.path.join(setDir, '*.txt'))):
            # wordFrameNumbers
            wordFrameNumbers = extract_word_frame_numbers(wordFileName, verbose=True)
            # images
            for jpgName in sorted(glob.glob('.'.join(wordFileName.split('.')[:-1]) + '*.jpg')):
                # mouth images
                if "mouth.jpg" in jpgName:
                    frameNumber = int(jpgName.split('/')[-1].split('.')[0].split('_')[-1])
                    # within word
                    if frameNumber in wordFrameNumbers:
                        print(jpgName)


