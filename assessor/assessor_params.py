import os
import sys

#############################################################
# IMPORT FROM OTHERS
#############################################################

LRW_ASSESSOR_DIR = os.path.dirname(os.path.realpath(__file__))

if LRW_ASSESSOR_DIR not in sys.path:
    sys.path.append(LRW_ASSESSOR_DIR)

LRW_DIR = os.path.realpath(os.path.join(LRW_ASSESSOR_DIR, '..'))

if LRW_DIR not in sys.path:
    sys.path.append(LRW_DIR)

LRW_HEAD_POSE_DIR = os.path.realpath(os.path.join(LRW_ASSESSOR_DIR, '../head-pose'))

if LRW_HEAD_POSE_DIR not in sys.path:
    sys.path.append(LRW_HEAD_POSE_DIR)

# PROCESS_LRW_DIR = os.path.normpath(os.path.join(ASSESSOR_DIR, "../process-lrw"))

# if PROCESS_LRW_DIR not in sys.path:
#     sys.path.append(PROCESS_LRW_DIR)

# from process_lrw_functions import *

#############################################################
# PARAMS
#############################################################

if 'voletiv' in os.getcwd():
    # voletiv
    LRW_DATA_DIR = '/home/voletiv/Datasets/LRW/lipread_mp4/'
    LRW_SAVE_DIR = '.'
    LRW_HEAD_POSE_DIR = '/home/voletiv/GitHubRepos/lipreading-in-the-wild-experiments/head-pose'
    GAZR_BUILD_DIR = '/home/voletiv/GitHubRepos/gazr/build'
    DEEPGAZE_EXAMPLES_DIR = '/home/voletiv/GitHubRepos/deepgaze/examples'
    SHAPE_DAT_FILE = "/home/voletiv/GitHubRepos/lipreading-in-the-wild-experiments/shape-predictor/shape_predictor_68_face_landmarks.dat"
elif 'voleti.vikram' in os.getcwd():
    # fusor
    LRW_DATA_DIR = '/shared/fusor/home/voleti.vikram/LRW-mouths'
    LRW_SAVE_DIR = '/shared/fusor/home/voleti.vikram/LRW-mouths'
    LRW_HEAD_POSE_DIR = '/shared/fusor/home/voleti.vikram/lipreading-in-the-wild-experiments/head-pose'
    GAZR_BUILD_DIR = '/shared/fusor/home/voleti.vikram/gazr/build'
    DEEPGAZE_EXAMPLES_DIR = '/shared/fusor/home/voleti.vikram/deepgaze/examples'
    SHAPE_DAT_FILE = "/shared/fusor/home/voleti.vikram/shape_predictor_68_face_landmarks.dat"

VIDEO_FPS = 25
VIDEO_FRAMES_PER_WORD = 30
MAX_FRAMES_PER_WORD = 21

N_OF_FRAMES_PER_SAMPLE_CSV_FILE = 'n_of_frames_per_sample.csv'
START_FRAMES_PER_SAMPLE_CSV_FILE = 'start_frames_per_sample.csv'

TIME_STEPS = MAX_FRAMES_PER_WORD
MOUTH_H = 112
MOUTH_W = 112
MOUTH_CHANNELS = 3

#############################################################
# LOAD VOCAB LIST
#############################################################


def load_lrw_vocab_list(LRW_VOCAB_LIST_FILE):
    lrw_vocab = []
    with open(LRW_VOCAB_LIST_FILE) as f:
        for line in f:
            word = line.rstrip().split()[-1]
            lrw_vocab.append(word)
    return lrw_vocab

LRW_VOCAB_LIST_FILE = os.path.join(LRW_DIR, 'lrw_vocabulary.txt')

LRW_VOCAB = load_lrw_vocab_list(LRW_VOCAB_LIST_FILE)

LRW_VOCAB_SIZE = len(LRW_VOCAB)
