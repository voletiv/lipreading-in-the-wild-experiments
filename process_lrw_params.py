print("Importing stuff...")

import dlib
import glob
# stackoverflow.com/questions/29718238/how-to-read-mp4-video-to-be-processed-by-scikit-image
import imageio
# import matplotlib
# matplotlib.use('agg')     # Use this for remote terminals, with ssh -X
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import subprocess
import tqdm
import warnings

from matplotlib.patches import Rectangle
from skimage.transform import resize

# Facial landmark detection
# http://dlib.net/face_landmark_detection.py.html

print("Done importing stuff.")

# # To ignore the deprecation warning from scikit-image
warnings.filterwarnings("ignore",category=UserWarning)

#############################################################
# PARAMS
#############################################################

if 'voletiv' in os.getcwd():
    # voletiv
    LRW_DATA_DIR = '/media/voletiv/01D2BF774AC76280/Datasets/LRW/lipread_mp4/'
    LRW_SAVE_DIR = '/home/voletiv/Datasets/LRW/lipread_mp4'

elif 'voleti.vikram' in os.getcwd():
    # fusor
    LRW_DATA_DIR = '/shared/magnetar/datasets/LipReading/LRW/lipread_mp4/'
    LRW_SAVE_DIR = '/shared/fusor/home/voleti.vikram/LRW-mouths'

#############################################################
# CONSTANTS
#############################################################

SHAPE_PREDICTOR_PATH = 'shape-predictor/shape_predictor_68_face_landmarks.dat'

FACIAL_LANDMARKS_IDXS = dict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])

MOUTH_SHAPE_FROM = FACIAL_LANDMARKS_IDXS["mouth"][0]
MOUTH_SHAPE_TO = FACIAL_LANDMARKS_IDXS["mouth"][1]

# Examples
videoFile = 'media/voletiv/01D2BF774AC76280/Datasets/LRW/lipread_mp4/ABOUT/test/ABOUT_00001.mp4'
wordFileName = '/home/voletiv/Datasets/LRW/lipread_mp4/ABOUT/test/ABOUT_00001.txt'
wordFileName = '/media/voletiv/01D2BF774AC76280/Datasets/LRW/lipread_mp4/ABOUT/test/ABOUT_00001.txt'
wordFileName = '/shared/magnetar/datasets/LipReading/LRW/lipread_mp4/ABOUT/test/ABOUT_00001.txt'
