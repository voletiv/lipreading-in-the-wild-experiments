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
import subprocess
import tqdm
import warnings

from matplotlib.patches import Rectangle
from skimage.transform import resize

# Facial landmark detection
# http://dlib.net/face_landmark_detection.py.html

print("Done importing stuff.")

# # To ignore the deprecation warning from scikit-image
# warnings.filterwarnings("ignore",category=UserWarning)

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

# process_lrw(rootDir=LRW_DATA_DIR, startExtracting=False,startSetWordNumber='val/ABOUT_00035',endSetWordNumber='val/ABOUT_00038',extractAudioFromMp4=True,dontWriteAudioIfExists=True,extractFramesFromMp4=False,writeFrameImages=False,dontWriteFrameIfExists=True,detectAndSaveMouths=False,dontWriteMouthIfExists=True)


# extract_and_save_audio_frames_and_mouths_from_dir
def process_lrw(rootDir=LRW_DATA_DIR,
                startExtracting=False,
                startSetWordNumber='train/ABOUT_00035',
                endSetWordNumber=None,
                extractAudioFromMp4=False,
                dontWriteAudioIfExists=True,
                extractFramesFromMp4=False,
                writeFrameImages=False,
                dontWriteFrameIfExists=True,
                detectAndSaveMouths=False,
                dontWriteMouthIfExists=True):

    # If something needs to be done:
    if extractAudioFromMp4 is False and \
            extractFramesFromMp4 is False and \
            detectAndSaveMouths is False:
        print("Nothing to be done!!\nAll extractAudioFromMp4, extractFramesFromMp4, detectAndSaveMouths are False!")
        return

    # Load detector and predictor if mouth is to be detected
    if detectAndSaveMouths:
        detector, predictor = load_detector_and_predictor()
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

                # Extract
                if startExtracting:
                    print(wordFileName)

                    # If endSetWordNumber is reached, end
                    if endSetWordNumber is not None:
                        if endSetWordNumber in wordFileName:
                            return

                    # Extract audio
                    if extractAudioFromMp4:
                        extractReturn = extract_audio_from_mp4(wordFileName, dontWriteAudioIfExists)

                    if extractFramesFromMp4 is False and detectAndSaveMouths is False:
                        continue

                    # Handling OSError in extracting frames and mouths
                    def please_extract(videoFile):
                        try:
                            # Extract frames and mouths
                            return extract_and_save_frames_and_mouths(wordFileName,
                                                                      extractFramesFromMp4,
                                                                      detectAndSaveMouths,
                                                                      writeFrameImages,
                                                                      dontWriteFrameIfExists,
                                                                      dontWriteMouthIfExists,
                                                                      detector,
                                                                      predictor)
                        except OSError:
                            print("Trying again...")
                            return please_extract(videoFile)
                        except KeyboardInterrupt:
                            print("Ctrl+C was pressed!")
                            return -1

                    # Extracting
                    extractReturn = please_extract(videoFile)
                    if extractReturn == -1:
                        return

#############################################################
# RUN ON ONE IMAGE
#############################################################


def test_mouth_detection_in_frame(rootDir=LRW_SAVE_DIR, word="ABOUT", set="train", number=1, frameNumber=1,
                         scaleFactor=.6, showMouthOnFrame=True, showResizedMouth=True,
                         detector=None, predictor=None,):

    if detector is None or predictor is None:
        detector, predictor = load_detector_and_predictor()

    # Make wordFileName
    wordFileName = os.path.join(
        rootDir, word, set, word + '_{0:05d}'.format(number) + '.txt')

    frame = read_jpeg_frames_from_dir(wordFileName)[frameNumber]
    face = detector(frame, 1)[0]
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


def load_detector_and_predictor():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    return detector, predictor


def extract_audio_from_mp4(wordFileName, dontWriteAudioIfExists):
    # Names
    videoFileName = '.'.join(wordFileName.split('.')[:-1]) + '.mp4'
    audioFileName = os.path.join(LRW_SAVE_DIR, "/".join(wordFileName.split("/")[-3:]).split('.')[0] + ".aac")
    # To overwrite or not to overwrite
    if dontWriteAudioIfExists:
        overwriteCommand = '-n'
    else:
        overwriteCommand = '-y'
    # Command
    command = ["ffmpeg", "-loglevel", "error", "-i", videoFileName, "-vn", overwriteCommand, "-acodec", "copy", audioFileName]
    return subprocess.call(command)



def extract_and_save_frames_and_mouths(
        wordFileName='/home/voletiv/Datasets/LRW/lipread_mp4/ABOUT/test/ABOUT_00001.txt',
        extractFramesFromMp4=False,
        writeFrameImages=False,
        detectAndSaveMouths=False,
        dontWriteFrameIfExists=True,
        dontWriteMouthIfExists=True,
        detector=None,
        predictor=None):
    # extractFramesFromMp4 and detectAndSaveMouths => Read frames from mp4 video and detect mouths
    # (not extractFramesFromMp4) and detectAndSaveMouths => Read frames from jpeg images and detect mouths
    # extractFramesFromMp4 and (not detectAndSaveMouths) => Read frames from mp4 video
    # (to maybe save them)

    # If extract frames from mp4 video
    if extractFramesFromMp4:
        videoFrames = extract_frames_from_video(wordFileName)
    # Else, read frame names in directory
    elif detectAndSaveMouths:
        videoFrames = read_jpeg_frames_from_dir(wordFileName)
    
    # Default face bounding box
    if detectAndSaveMouths:
        face = dlib.rectangle(30, 30, 220, 220)

    # For each frame
    for f, frame in enumerate(videoFrames):

        # Write the frame image (from video)
        if extractFramesFromMp4 and writeFrameImages:

            frameImageName = os.path.join(LRW_SAVE_DIR, "/".join(wordFileName.split(
                "/")[-3:]).split('.')[0] + "_{0:02d}".format(f + 1) + ".jpg")
            
            # If file is not supposed to be written if it exists
            if dontWriteFrameIfExists:
                if not os.path.isfile(frameImageName):
                    imageio.imwrite(frameImageName, frame)
            else:
                imageio.imwrite(frameImageName, frame)

        # Detect mouths in frames
        if detectAndSaveMouths:
            
            # Image Name
            mouthImageName = os.path.join(LRW_SAVE_DIR, "/".join(wordFileName.split(
                                          "/")[-3:]).split('.')[0] + \
                                          "_{0:02d}_mouth".format(f + 1) + ".jpg")
            
            # If file is not supposed to be written if it exists
            if dontWriteMouthIfExists:
                if os.path.isfile(mouthImageName):
                    continue

            if detector is None or predictor is None:
                print("Please specify dlib detector/predictor!!")
                return -1

            # Detect and save mouth in frame
            mouthImage, face = detect_mouth_in_frame(frame, detector,
                                                     predictor,
                                                     prevFace=face)

            # Save mouth image
            imageio.imwrite(mouthImageName, mouthImage)

    return 1


def extract_frames_from_video(wordFileName):
    videoFileName = '.'.join(wordFileName.split('.')[:-1]) + '.mp4'
    videoFrames = imageio.get_reader(videoFileName, 'ffmpeg')
    return videoFrames


def read_jpeg_frames_from_dir(wordFileName):
    # Frame names end with numbers from 00 to 30, so [0-3][0-9]
    videoFrameNames = sorted(
        glob.glob(os.path.join(LRW_SAVE_DIR,
                               "/".join(wordFileName.split("/")[-3:]).split('.')[0] + \
                               '_[0-3][0-9].jpg')))
    # Read all frame images
    videoFrames = []
    for frameName in videoFrameNames:
        videoFrames.append(imageio.imread(frameName))
    # Return
    return videoFrames


def detect_mouth_in_frame(frame, detector, predictor,
                          prevFace=dlib.rectangle(30, 30, 220, 220)):
    # Shape Coords: ------> x (cols)
    #               |
    #               |
    #               v
    #               y
    #             (rows)

    # Detect face
    face = detector(frame, 1)
    if len(face) == 0:
        face = prevFace
    else:
        face = face[0]

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
