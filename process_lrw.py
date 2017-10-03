import dlib
import glob
# stackoverflow.com/questions/29718238/how-to-read-mp4-video-to-be-processed-by-scikit-image
import imageio
# import matplotlib
# matplotlib.use('agg')     # Use this for remote terminals
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm

from matplotlib.patches import Rectangle
from skimage.transform import resize

# Facial landmark detection
# http://dlib.net/face_landmark_detection.py.html

#############################################################
# PARAMS
#############################################################

LRW_DATA_DIR = '/media/voletiv/01D2BF774AC76280/Datasets/LRW/lipread_mp4/'
LRW_SAVE_DIR = '/home/voletiv/Datasets/LRW/lipread_mp4'
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

#############################################################
# EXTRACT FRAMES AND MOUTHS
#############################################################


def extract_and_save_frames_and_mouths_from_dir(rootDir=LRW_DATA_DIR,
                                                startExtracting=False,
                                                startDir='train/ABOUT_00035',
                                                extractFrames=False,
                                                detectAndSaveMouths=False,
                                                writeFrameImages=False,
                                                dontWriteFrameIfExists=True,
                                                dontWriteMouthIfExists=True,
                                                mouthW=112):
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

                # Don't extract until all previously extract_return are passed
                if startDir in wordFileName:
                    startExtracting = True

                # Extract
                if startExtracting:
                    if detectAndSaveMouths:
                        detector = dlib.get_frontal_face_detector()
                        predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

                    # Handling OSError
                    def please_extract(videoFile):
                        try:
                            # Extract frames and mouths
                            return extract_and_save_frames_and_mouths(wordFileName,
                                                                      extractFrames,
                                                                      detectAndSaveMouths,
                                                                      writeFrameImages,
                                                                      dontWriteFrameIfExists,
                                                                      dontWriteMouthIfExists,
                                                                      detector,
                                                                      predictor,
                                                                      mouthW)
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


def extract_and_save_frames_and_mouths(
        wordFileName='/home/voletiv/Datasets/LRW/lipread_mp4/ABOUT/test/ABOUT_00001.txt',
        extractFrames=False,
        detectAndSaveMouths=False,
        writeFrameImages=False,
        dontWriteFrameIfExists=True,
        dontWriteMouthIfExists=True,
        detector=None,
        predictor=None,
        mouthW=112):
    # extractFrames and detectAndSaveMouths => Read frames from mp4 video and detect mouths
    # (not extractFrames) and detectAndSaveMouths => Read frames from jpeg images and detect mouths
    # extractFrames and (not detectAndSaveMouths) => Read frames from mp4 video
    # (to maybe save them)

    # If something needs to be done:
    if extractFrames or detectAndSaveMouths:

        # If extract frames from mp4 video
        if extractFrames:
            videoFrames = extract_frames_from_video(wordFileName)

        # Else, read frame names in folder
        elif detectAndSaveMouths:
            videoFrames = read_jpeg_frames_from_dir(wordFileName)

        # For each frame
        for f, frame in enumerate(videoFrames):

            # Write the frame image (from video)
            if extractFrames and writeFrameImages:
                write_lrw_image(wordFileName, f, frame, mouth=False,
                                dontWriteIfExists=dontWriteFrameIfExists)

            # Detect mouths in frames
            if detectAndSaveMouths:

                if detector is None or predictor is None:
                    print("Please specify dlib detector/predictor!!")
                    return -1

                # Detect and save mouth in frame
                mouthImage = detect_mouth_in_frame(frame, detector, predictor)

                # Save mouth image
                write_lrw_image(wordFileName, f, mouthImage, mouth=True,
                                dontWriteIfExists=dontWriteMouthIfExists)

    return 1


def extract_frames_from_video(wordFileName):
    videoFileName = '.'.join(wordFileName.split('.')[:-1]) + '.mp4'
    videoFrames = imageio.get_reader(videoFileName, 'ffmpeg')
    return videoFrames


def read_jpeg_frames_from_dir(wordFileName):
    # Frame names end with numbers from 00 to 30, so [0-3][0-9]
    videoFrameNames = sorted(
        glob.glob('.'.join(wordFileName.split('.')[:-1]) + '_[0-3][0-9].jpg'))
    # Read all frame images
    videoFrames = []
    for frameName in videoFrameNames:
        videoFrames.append(imageio.imread(frameName))
    # Return
    return videoFrames


def write_lrw_image(wordFileName, f, image, mouth=False, dontWriteIfExists=True):
    # Note the name of file to be saved
    if not mouth:
        imageName = os.path.join(LRW_SAVE_DIR, "/".join(wordFileName.split(
            "/")[-3:]).split('.')[0] + "_{0:02d}".format(f + 1) + ".jpg")
    else:
        imageName = os.path.join(LRW_SAVE_DIR, "/".join(wordFileName.split(
            "/")[-3:]).split('.')[0] + "_{0:02d}_mouth".format(f + 1) + ".jpg")
    # print(imageName)

    # Save if file doesn't exist
    if dontWriteIfExists:
        if not os.path.isfile(imageName):
            imageio.imwrite(imageName, image)
    else:
        imageio.imwrite(imageName, image)


def detect_mouth_in_frame(frame, detector, predictor):
    # Shape Coords: ------> x (cols)
    #               |
    #               |
    #               v
    #               y
    #             (rows)

    # Detect face
    face = detector(frame, 1)[0]

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
    return resizedMouthImage


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


def test_mouth_detection(word="ABOUT", set="train", number=1, frameNumber=1,
                         scaleFactor=.5, showMouthOnFrame=True, showResizedMouth=True):
    wordFileName = os.path.join(
        LRW_SAVE_DIR, word, set, word + '_{0:05d}'.format(number) + '.txt')

    frame = read_jpeg_frames_from_dir(wordFileName)[frameNumber]
    face = detector(frame, 1)[0]
    shape = predictor(frame, face)

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
