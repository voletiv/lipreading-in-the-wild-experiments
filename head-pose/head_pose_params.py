import os
import sys

HEAD_POSE_DIR = os.path.dirname(os.path.realpath(__file__))

# if LRW_DIR not in sys.path:
#     sys.path.append(LRW_DIR)

if 'voletiv' in os.getcwd():
    # voletiv
    LRW_DATA_DIR = '/home/voletiv/Datasets/LRW/lipread_mp4/'
    LRW_SAVE_DIR = '.'
    GAZR_BUILD_DIR = '/home/voletiv/GitHubRepos/gazr/build'
    SHAPE_DAT_FILE = "/home/voletiv/GitHubRepos/lipreading-in-the-wild-experiments/shape-predictor/shape_predictor_68_face_landmarks.dat"
elif 'voleti.vikram' in os.getcwd():
    # fusor
    LRW_DATA_DIR = '/shared/fusor/home/voleti.vikram/LRW-mouths'
    LRW_SAVE_DIR = '/shared/fusor/home/voleti.vikram/LRW-mouths'
    GAZR_BUILD_DIR = '/shared/fusor/home/voleti.vikram/gazr/build'
    SHAPE_DAT_FILE = "/shared/fusor/home/voleti.vikram/lipreading-in-the-wild-experiments/shape-predictor/shape_predictor_68_face_landmarks.dat"
