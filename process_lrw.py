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

#############################################################
# EXTRACT AUDIO, FRAMES, AND MOUTHS
#############################################################

# process_lrw(rootDir=LRW_DATA_DIR,startExtracting=True,startSetWordNumber='test/ALWAYS_00057',endSetWordNumber=None,copyTxtFile=False,extractAudioFromMp4=True,dontWriteAudioIfExists=True,extractFramesFromMp4=True,writeFrameImages=True,dontWriteFrameIfExists=True,detectAndSaveMouths=True,dontWriteMouthIfExists=True,verbose=False)


# extract_and_save_audio_frames_and_mouths_from_dir
def process_lrw(rootDir=LRW_DATA_DIR,
                startExtracting=False,
                startSetWordNumber='train/ABOUT_00035',
                endSetWordNumber=None,
                copyTxtFile=False,
                extractAudioFromMp4=False,
                dontWriteAudioIfExists=True,
                extractFramesFromMp4=False,
                writeFrameImages=False,
                dontWriteFrameIfExists=True,
                detectAndSaveMouths=False,
                dontWriteMouthIfExists=True,
                verbose=False):

    # If nothing needs to be done:
    if copyTxtFile is False and \
            extractAudioFromMp4 is False and \
            extractFramesFromMp4 is False and \
            detectAndSaveMouths is False:
        print("Nothing to be done!!\nAll coptTxtFile, extractAudioFromMp4, extractFramesFromMp4, detectAndSaveMouths are False!")
        return

    # If mouth is to be detected, Load detector and predictor
    if detectAndSaveMouths:
        try:
            detector, predictor = load_detector_and_predictor(verbose)
        # If SHAPE_PREDICTOR_PATH is wrong
        except ValueError as err:
            print(err)
            return
        # Ctrl+C
        except KeyboardInterrupt:
            print("Ctrl+C was pressed!")
            return
    else:
        detector = None
        predictor = None

    # For each word
    for wordDir in tqdm.tqdm(sorted(glob.glob(os.path.join(rootDir, '*/')))):
        print(wordDir)

        # train, val or test
        for setDir in tqdm.tqdm(sorted(glob.glob(os.path.join(wordDir, '*/')))):
            print(setDir)

            # Read all .txt file names (since .txt are saved in both
            # LRW_DATA_DIR and LRW_SAVE_DIR)
            wordFileNames = sorted(glob.glob(os.path.join(setDir, '*.txt')))

            # For each video
            for wordFileName in tqdm.tqdm(wordFileNames):

                # Don't extract until startSetWordNumber is reached
                if startSetWordNumber in wordFileName:
                    startExtracting = True

                if not startExtracting:
                    continue

                print(wordFileName)

                # If endSetWordNumber is reached, end
                if endSetWordNumber is not None:
                    if endSetWordNumber in wordFileName:
                        return

                # Copy .txt file containing word duration info
                if copyTxtFile:
                    copyTxtFile(wordFileName, verbose)

                # Extract audio
                if extractAudioFromMp4:
                    try:
                        extractReturn = extract_audio_from_mp4(wordFileName=wordFileName,
                            dontWriteAudioIfExists=dontWriteAudioIfExists, verbose=verbose)

                    except KeyboardInterrupt:
                        print("Ctrl+C was pressed!")
                        return

                if extractFramesFromMp4 is False and detectAndSaveMouths is False:
                    continue

                # Handling Memory I/O Error (OSError) in extracting frames and mouths
                def please_extract(videoFile):
                    try:
                        # Extract frames and mouths
                        return extract_and_save_frames_and_mouths(wordFileName=wordFileName,
                                                                  extractFramesFromMp4=extractFramesFromMp4,
                                                                  writeFrameImages=writeFrameImages,
                                                                  detectAndSaveMouths=detectAndSaveMouths,
                                                                  dontWriteFrameIfExists=dontWriteFrameIfExists,
                                                                  dontWriteMouthIfExists=dontWriteMouthIfExists,
                                                                  detector=detector,
                                                                  predictor=predictor,
                                                                  verbose=verbose)

                    # Memory I/O error
                    except OSError:
                        print("Trying again...")
                        return please_extract(videoFile)
                    
                    # Ctrl+C
                    except KeyboardInterrupt:
                        print("Ctrl+C was pressed!\n\n")
                        return 1

                # Extracting
                extractReturn = please_extract(videoFile)
                if extractReturn == 1:
                    return

#############################################################
# RUN ON ONE IMAGE
#############################################################


def test_mouth_detection_in_frame(rootDir=LRW_SAVE_DIR, word="ABOUT",
                                  set="train", number=1, frameNumber=1,
                                  scaleFactor=.6, showMouthOnFrame=True,
                                  showResizedMouth=True, detector=None,
                                  predictor=None, verbose=False):

    if detector is None or predictor is None:
        try:
            detector, predictor = load_detector_and_predictor(verbose)
        except ValueError as err:
            print(err)
            return

    # Make wordFileName
    wordFileName = os.path.join(
        rootDir, word, set, word + '_{0:05d}'.format(number) + '.txt')

    frame = read_jpeg_frames_from_dir(wordFileName)[frameNumber]
    try:
        face = detector(frame, 1)[0]
    except IndexError:
        shape = predictor(frame, face)

    # # Show landmarks and face
    # win = dlib.image_window()
    # win.set_image(frame)
    # win.add_overlay(shape)
    # win.add_overlay(face)

    mouthCoords = np.array([[shape.part(i).x, shape.part(i).y]
                            for i in range(MOUTH_SHAPE_FROM, MOUTH_SHAPE_TO)])
    mouthRect = (np.min(mouthCoords[:, 0]), np.min(mouthCoords[:, 1]),
                 np.max(mouthCoords[:, 0]) - np.min(mouthCoords[:, 0]),
                 np.max(mouthCoords[:, 1]) - np.min(mouthCoords[:, 1]))
    mouthRect = make_rect_shape_square(mouthRect)

    scale = scaleFactor * face.width() / mouthRect[2]
    # print("scale =", scale)
    croppedScale = 112 / 120 * scale

    expandedMouthRect = expand_rect(mouthRect, scale=scale)
    expandedCroppedMouthRect = expand_rect(mouthRect, scale=croppedScale)

    if showMouthOnFrame:
        plt.subplot(121)
        plt.imshow(frame)
        ca = plt.gca()
        ca.add_patch(Rectangle((expandedMouthRect[0], expandedMouthRect[1]),
                               expandedMouthRect[2], expandedMouthRect[3],
                               edgecolor='r', fill=False))
        ca.add_patch(Rectangle((expandedCroppedMouthRect[0],
                                expandedCroppedMouthRect[1]),
                               expandedCroppedMouthRect[2],
                               expandedCroppedMouthRect[3],
                               edgecolor='g', fill=False))

    if showResizedMouth:
        resizedMouthImage \
            = np.round(resize(frame[expandedMouthRect[1]:expandedMouthRect[1] + expandedMouthRect[3],
                                    expandedMouthRect[0]:expandedMouthRect[0] + expandedMouthRect[2]],
                              (120, 120), preserve_range=True)).astype('uint8')
        plt.subplot(122)
        plt.imshow(resizedMouthImage)

    if showMouthOnFrame or showResizedMouth:
        plt.show()

    return wordFileName, resizedMouthImage

#############################################################
# DEPENDENT FUNCTIONS
#############################################################


def load_detector_and_predictor(verbose=False):
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        if verbose:
            print("Detector and Predictor loaded.")
        return detector, predictor
    except RuntimeError:
        raise ValueError("ERROR: Please specify Shape Predictor .dat file full path correctly!!\nSpecified path = " + \
            SHAPE_PREDICTOR_PATH)


def copyTxtFile(wordFileName, verbose=False):
    return


def extract_audio_from_mp4(wordFileName, dontWriteAudioIfExists, verbose=False):
    # Names
    videoFileName = '.'.join(wordFileName.split('.')[:-1]) + '.mp4'
    audioFileName = os.path.join(LRW_SAVE_DIR,
        "/".join(wordFileName.split("/")[-3:]).split('.')[0] + ".aac")
    
    # Don't write if .aac file exists
    if dontWriteAudioIfExists:
        if os.path.isfile(audioFileName):
            if verbose:
                print("Audio file", audioFileName, "exists, so not written")
            return 0
    
    # Just in case, to overwrite or not to overwrite
    if dontWriteAudioIfExists:
        overwriteCommand = '-n'
    else:
        overwriteCommand = '-y'
    
    # Command
    command = ["ffmpeg", "-loglevel", "error", "-i", videoFileName, "-vn",
               overwriteCommand, "-acodec", "copy", audioFileName]

    # subprocess.call returns 0 on successful run
    extractReturn = subprocess.call(command)

    if verbose:
        if extractReturn == 0:
            print("Audio file written.")
        else:
            print("ERROR: Audio file NOT WRITEN!!")

    return extractReturn


def extract_and_save_frames_and_mouths(
        wordFileName='/home/voletiv/Datasets/LRW/lipread_mp4/ABOUT/test/ABOUT_00001.txt',
        extractFramesFromMp4=False,
        writeFrameImages=False,
        detectAndSaveMouths=False,
        dontWriteFrameIfExists=True,
        dontWriteMouthIfExists=True,
        detector=None,
        predictor=None,
        verbose=False):
    # extractFramesFromMp4 and detectAndSaveMouths => Read frames from mp4 video and detect mouths
    # (not extractFramesFromMp4) and detectAndSaveMouths => Read frames from jpeg images and detect mouths
    # extractFramesFromMp4 and (not detectAndSaveMouths) => Read frames from mp4 video
    # (to maybe save them)

    # If extract frames from mp4 video
    if extractFramesFromMp4:
        videoFrames = extract_frames_from_video(wordFileName, verbose)
    
    # Else, read frame names in directory
    elif detectAndSaveMouths:
        videoFrames = read_jpeg_frames_from_dir(wordFileName, verbose)

    # Default face bounding box
    if detectAndSaveMouths:
        # Default face
        face = dlib.rectangle(30, 30, 220, 220)

    # For each frame
    for f, frame in enumerate(videoFrames):

        # Write the frame image (from video)
        if extractFramesFromMp4 and writeFrameImages:
            write_frame_image(wordFileName, f, frame, dontWriteFrameIfExists,
                verbose)
            
        # Detect mouths in frames
        if detectAndSaveMouths:
            face = detect_mouth_and_write(wordFileName, f, frame, detector, predictor,
                dontWriteMouthIfExists, prevFace=face, verbose=verbose)

    return 0


def extract_frames_from_video(wordFileName, verbose=False):
    videoFileName = '.'.join(wordFileName.split('.')[:-1]) + '.mp4'
    videoFrames = imageio.get_reader(videoFileName, 'ffmpeg')
    if verbose:
            print("Frames extracted from video")
    # Return
    return videoFrames


def read_jpeg_frames_from_dir(wordFileName, verbose=False):
    # Frame names end with numbers from 00 to 30, so [0-3][0-9]
    videoFrameNames = sorted(
        glob.glob(os.path.join(LRW_SAVE_DIR,
                               "/".join(wordFileName.split("/")[-3:]).split('.')[0] + \
                               '_[0-3][0-9].jpg')))
    # Read all frame images
    videoFrames = []
    for frameName in videoFrameNames:
        videoFrames.append(imageio.imread(frameName))
    # Print
    if verbose:
            print("Frames read from jpeg images")
    # Return
    return videoFrames


def write_frame_image(wordFileName, f, frame, dontWriteFrameIfExists=True,
        verbose=False):
    # Name
    frameImageName = os.path.join(LRW_SAVE_DIR, "/".join(wordFileName.split(
        "/")[-3:]).split('.')[0] + "_{0:02d}".format(f + 1) + ".jpg")
    
    # If file is not supposed to be written if it exists
    if dontWriteFrameIfExists:
        if not os.path.isfile(frameImageName):
            imageio.imwrite(frameImageName, frame)
            if verbose:
                print("Frame image", frameImageName, "written")
        else:
            if verbose:
                print("Frame image", frameImageName, "exists, so not written")
    else:
        imageio.imwrite(frameImageName, frame)
        if verbose:
            print("Frame image", frameImageName, "written")


def detect_mouth_and_write(wordFileName, f, frame, detector, predictor,
        dontWriteMouthIfExists=True, prevFace=dlib.rectangle(30, 30, 220, 220),
        verbose=False):
    # Image Name
    mouthImageName = os.path.join(LRW_SAVE_DIR, "/".join(wordFileName.split(
                                  "/")[-3:]).split('.')[0] + \
                                  "_{0:02d}_mouth".format(f + 1) + ".jpg")
    
    # If file is not supposed to be written if it exists
    if dontWriteMouthIfExists:
        if os.path.isfile(mouthImageName):
            if verbose:
                print("Mouth image", mouthImageName, "exists, so not detected")
            return face

    # Detect and save mouth in frame
    mouthImage, face = detect_mouth_in_frame(frame, detector, predictor,
                                             prevFace=prevFace)

    # Save mouth image
    imageio.imwrite(mouthImageName, mouthImage)
    if verbose:
        print("Mouth image", mouthImageName, "written")

    # Return
    return face


def detect_mouth_in_frame(frame, detector, predictor,
                          prevFace=dlib.rectangle(30, 30, 220, 220)):
    # Shape Coords: ------> x (cols)
    #               |
    #               |
    #               v
    #               y
    #             (rows)

    # Detect face
    try:
        face = detector(frame, 1)[0]
    except IndexError:
        face = prevFace

    # Predict facial landmarks
    shape = predictor(frame, face)

    # # Show landmarks and face
    # win = dlib.image_window()
    # win.set_image(frame)
    # win.add_overlay(shape)
    # win.add_overlay(face)

    # Note all mouth landmark coordinates
    mouthCoords = np.array([[shape.part(i).x, shape.part(i).y]
                            for i in range(MOUTH_SHAPE_FROM, MOUTH_SHAPE_TO)])

    # Mouth Rect: x, y, w, h
    mouthRect = (np.min(mouthCoords[:, 0]), np.min(mouthCoords[:, 1]),
                 np.max(mouthCoords[:, 0]) - np.min(mouthCoords[:, 0]),
                 np.max(mouthCoords[:, 1]) - np.min(mouthCoords[:, 1]))

    # Make mouthRect square
    mouthRect = make_rect_shape_square(mouthRect)

    # Expand mouthRect square
    expandedMouthRect = expand_rect(
        mouthRect, scale=(0.6 * face.width() / mouthRect[2]))

    # Resize to 120x120
    resizedMouthImage = np.round(resize(frame[expandedMouthRect[1]:expandedMouthRect[1] + expandedMouthRect[3],
                                              expandedMouthRect[0]:expandedMouthRect[0] + expandedMouthRect[2]],
                                        (120, 120), preserve_range=True)).astype('uint8')

    # Return mouth
    return resizedMouthImage, face


def make_rect_shape_square(rect):
    # Rect: (x, y, w, h)
    # If width > height
    if rect[2] > rect[3]:
        rect = (rect[0], int(rect[1] + rect[3] / 2 - rect[2] / 2),
                rect[2], rect[2])
    # Else (height > width)
    else:
        rect = (int(rect[0] + rect[2] / 2 - rect[3] / 2), rect[1],
                rect[3], rect[3])
    # Return
    return rect


def expand_rect(rect, scale=1.5):
    # Rect: (x, y, w, h)
    w = int(rect[2] * scale)
    h = int(rect[3] * scale)
    x = rect[0] - int((w - rect[2]) / 2)
    y = rect[1] - int((h - rect[3]) / 2)
    return (x, y, w, h)
