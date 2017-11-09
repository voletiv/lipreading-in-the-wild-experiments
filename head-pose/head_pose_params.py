import os
import sys

#############################################################
# IMPORT FROM OTHERS
#############################################################

HEAD_POSE_DIR = os.path.dirname(os.path.realpath(__file__))

if HEAD_POSE_DIR not in sys.path:
    sys.path.append(HEAD_POSE_DIR)

# PROCESS_LRW_DIR = os.path.normpath(os.path.join(HEAD_POSE_DIR, "../process-lrw"))

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
    GAZR_BUILD_DIR = '/home/voletiv/GitHubRepos/gazr/build'
    DEEPGAZE_EXAMPLES_DIR = '/home/voletiv/GitHubRepos/deepgaze/examples'
    SHAPE_DAT_FILE = "/home/voletiv/GitHubRepos/lipreading-in-the-wild-experiments/shape-predictor/shape_predictor_68_face_landmarks.dat"
elif 'voleti.vikram' in os.getcwd():
    # fusor
    LRW_DATA_DIR = '/shared/fusor/home/voleti.vikram/LRW-mouths'
    LRW_SAVE_DIR = '/shared/fusor/home/voleti.vikram/LRW-mouths'
    GAZR_BUILD_DIR = '/shared/fusor/home/voleti.vikram/gazr/build'
    DEEPGAZE_EXAMPLES_DIR = '/shared/fusor/home/voleti.vikram/deepgaze/examples'
    SHAPE_DAT_FILE = "/shared/fusor/home/voleti.vikram/shape_predictor_68_face_landmarks.dat"

VIDEO_FPS = 25
VIDEO_FRAMES_PER_WORD = 30
