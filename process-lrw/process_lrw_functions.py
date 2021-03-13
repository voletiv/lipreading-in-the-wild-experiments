from __future__ import print_function

print("Importing stuff...")

import dlib
import glob
# stackoverflow.com/questions/29718238/how-to-read-mp4-video-to-be-processed-by-scikit-image
import imageio
import math
# import matplotlib
# matplotlib.use('agg')     # Use this for remote terminals, with ssh -X
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import subprocess
import time
import tqdm

from matplotlib.patches import Rectangle
from skimage.transform import resize

# Facial landmark detection
# http://dlib.net/face_landmark_detection.py.html

from process_lrw_params import *

print("Done importing stuff.")


#############################################################
# EXTRACT AUDIO, FRAMES, AND MOUTHS
#############################################################


# extract_and_save_audio_frames_and_mouths_from_dir
def process_lrw(dataDir=LRW_DATA_DIR,
                saveDir=LRW_SAVE_DIR,
                startExtracting=False,
                startSetWordNumber='train/ABOUT_00035',
                endSetWordNumber=None,
                copyTxtFile=False,
                extractAudioFromMp4=False,
                dontWriteAudioIfExists=True,
                extractFramesFromMp4=False,
                extractOnlyWordFrames=True,
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
        print("\n\nNothing to be done!!\nAll coptTxtFile, extractAudioFromMp4, extractFramesFromMp4, detectAndSaveMouths are False!\n\n")
        return

    # If dataDir is not valid
    if not os.path.isdir(dataDir):
        print("\n\nERROR: dataDir is not a valid directory:", dataDir, "\n\n")
        return

    # If saveDir is not valid
    if not os.path.isdir(saveDir):
        print("\n\nERROR: saveDir is not a valid directory:", saveDir, "\n\n")
        return

    # If startSetWordNumber is not valid
    if startExtracting is False:
        if not os.path.isfile(os.path.join(dataDir,
                startSetWordNumber.split('/')[1].split('_')[0],
                startSetWordNumber) + '.txt'):
            print("\n\nERROR: startSetWordNumber not valid:", startSetWordNumber, '\n\n')
            return

    # If endSetWordNumber is not valid
    if endSetWordNumber is not None:
        if not os.path.isfile(os.path.join(dataDir,
                endSetWordNumber.split('/')[1].split('_')[0],
                endSetWordNumber) + '.txt'):
            print("\n\nERROR: endSetWordNumber not valid:", endSetWordNumber, '\n\n')
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
            print("\n\nCtrl+C was pressed!\n\n")
            return
    else:
        detector = None
        predictor = None

    # Time
    start_time = time.time()

    # LOOP THROUGH ALL DIRECTORIES IN LRW DATASET
    # For each word
    for wordDir in tqdm.tqdm(sorted(glob.glob(os.path.join(dataDir, '*/')))):
        print(wordDir)
        print_time_till_now(start_time)

        # train, val or test
        for setDir in tqdm.tqdm(sorted(glob.glob(os.path.join(wordDir, '*/')))):
            print(setDir)
            print_time_till_now(start_time)

            # Create directory in saveDir if it doesn't exist
            setSaveDir = os.path.join(saveDir, '/'.join(os.path.normpath(setDir).split('/')[-2:]))
            if not os.path.isdir(setSaveDir):
                os.makedirs(setSaveDir)

            # Read all .txt file names (since .txt are saved in both
            # LRW_DATA_DIR and LRW_SAVE_DIR)
            # For each video
            for wordFileName in tqdm.tqdm(sorted(glob.glob(os.path.join(setDir, '*.txt')))):

                # Check if the directory to start at has been reached
                if startSetWordNumber in wordFileName:
                    startExtracting = True

                # Don't extract until startSetWordNumber is reached
                if not startExtracting:
                    continue

                print(wordFileName)
                print_time_till_now(start_time)

                # If endSetWordNumber is reached, end
                if endSetWordNumber is not None:
                    if endSetWordNumber in wordFileName:
                        return

                # Copy .txt file containing word duration info
                if copyTxtFile:
                    try:
                        copy_txt_file(saveDir, wordFileName, verbose)
                    except ValueError as err:
                        print(err)
                    except KeyboardInterrupt:
                        print("\n\nCtrl+C was pressed!\n\n")
                        return

                # Extract audio
                if extractAudioFromMp4:
                    try:
                        extract_audio_from_mp4(saveDir=saveDir, wordFileName=wordFileName,
                            dontWriteAudioIfExists=dontWriteAudioIfExists, verbose=verbose)
                    except ValueError as err:
                        print(err)
                    except KeyboardInterrupt:
                        print("\n\nCtrl+C was pressed!\n\n")
                        return

                # If frames don't need to be extracted from mp4 video,
                # and mouths don't need to be detected, continue
                if extractFramesFromMp4 is False and detectAndSaveMouths is False:
                    continue

                # Handling Memory I/O Error (OSError) in reading videos or
                # frames, or if files to be read are not present
                def please_extract(videoFile):
                    try:
                        # Extract frames and mouths
                        return extract_and_save_frames_and_mouths(saveDir=saveDir,
                                                                  wordFileName=wordFileName,
                                                                  extractFramesFromMp4=extractFramesFromMp4,
                                                                  extractOnlyWordFrames=extractOnlyWordFrames,
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

                    # File non-existence error
                    except ValueError as err:
                        print(err)
                        return 1

                    # Ctrl+C
                    except KeyboardInterrupt:
                        print("\n\nCtrl+C was pressed!\n\n")
                        return 1

                # Extracting
                extractReturn = please_extract(videoFile)

                # If Ctrl+C was pressed
                if extractReturn == 1:
                    return

#############################################################
# DEPENDENT FUNCTIONS
#############################################################


def print_time_till_now(start_time):
    ret = os.system("date")
    till_now = time.time() - start_time
    h = till_now//3600
    m = (till_now - h*3600)//60
    s = (till_now - h*3600 - m*60)//1
    print(h, "hr", m, "min", s, "sec")


def load_detector_and_predictor(verbose=False):
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        if verbose:
            print("Detector and Predictor loaded. (load_detector_and_predictor)")
        return detector, predictor
    # If error in SHAPE_PREDICTOR_PATH
    except RuntimeError:
        raise ValueError("\n\nERROR: Wrong Shape Predictor .dat file path - " + \
            SHAPE_PREDICTOR_PATH, "(load_detector_and_predictor)\n\n")


def copy_txt_file(saveDir, wordFileName, verbose=False):
    # Names
    fromFileName = wordFileName
    toFileName = os.path.join(saveDir, "/".join(wordFileName.split("/")[-3:]))
    try:
        shutil.copyfile(fromFileName, toFileName)
        if verbose:
            print("Text file copied:", fromFileName, "->", toFileName,
                "(copy_txt_file)")
        return 0
    except:
        raise ValueError("\n\nERROR: shutil failed to copy " + fromFileName + \
            " to " + toFileName + " (copy_txt_file)\n\n")


def extract_audio_from_mp4(saveDir, wordFileName, dontWriteAudioIfExists, verbose=False):
    # Names
    videoFileName = '.'.join(wordFileName.split('.')[:-1]) + '.mp4'
    audioFileName = os.path.join(saveDir,
        "/".join(wordFileName.split("/")[-3:]).split('.')[0] + ".aac")
    
    # Don't write if .aac file exists
    if dontWriteAudioIfExists:
        # Check if file exists
        if os.path.isfile(audioFileName):
            if verbose:
                print("Audio file, exists, so not written:" + audioFileName + \
                    " (extract_audio_from_mp4)")
            # Return if file exists
            return

    # Just in case, to overwrite or not to overwrite
    if dontWriteAudioIfExists:
        overwriteCommand = '-n'
    else:
        overwriteCommand = '-y'

    # Command
    command = ["ffmpeg", "-loglevel", "error", "-i", videoFileName, "-vn",
               overwriteCommand, "-acodec", "copy", audioFileName]

    # subprocess.call returns 0 on successful run
    try:
        commandReturn = subprocess.call(command)
    except KeyboardInterrupt:
        raise KeyboardInterrupt

    # If audio file could not be written by subprocess
    if commandReturn != 0:
        raise ValueError("\n\nERROR: Audio file " + audioFileName + " NOT WRITEN!! (extract_audio_from_mp4)\n\n")

    if verbose:
        if commandReturn == 0:
            print("Audio file written:", audioFileName, "(extract_audio_from_mp4)")


def extract_and_save_frames_and_mouths(saveDir=LRW_SAVE_DIR,
        wordFileName='/home/voletiv/Datasets/LRW/lipread_mp4/ABOUT/test/ABOUT_00001.txt',
        extractFramesFromMp4=False,
        extractOnlyWordFrames=True,
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

    try:
        # If extract frames from mp4 video
        if extractFramesFromMp4:
            videoFrames = extract_frames_from_video(wordFileName, verbose)
        # Else, read frame names in directory
        elif detectAndSaveMouths:
            videoFrames, videoFrameNames = read_jpeg_frames_from_dir(saveDir,
                wordFileName, verbose)
    # If mp4 or jpeg files to read are missing, cascade ValueError up
    except ValueError as err:
        raise ValueError(err)

    # Default face bounding box
    if detectAndSaveMouths:
        # Default face
        face = dlib.rectangle(30, 30, 220, 220)

    # If only the 5 or 6 frames with the word, given by the word duration
    # in the .txt file, are to be extracted
    if extractOnlyWordFrames:
        wordFrameNumbers = extract_word_frame_numbers(wordFileName, verbose=verbose)

    # For each frame
    for f, frame in enumerate(videoFrames):

        # If frames are extracted from video, all frames are read
        if extractFramesFromMp4:
            frameNumer = f + 1
        # If frames are read from jpeg images, frame numbers are in their names
        else:
            frameNumer = int(videoFrameNames[f].split('/')[-1].split('.')[0].split('_')[-1])

        # If only the 5 or 6 frames with the word, given by the word duration
        # in the .txt file, are to be extracted
        if extractOnlyWordFrames:
            # Extract only the wordFrameNumbers
            if frameNumer not in wordFrameNumbers:
                continue

        # Write the frame image (from video)
        if extractFramesFromMp4 and writeFrameImages:
            write_frame_image(saveDir=saveDir, wordFileName=wordFileName,
                frameNumer=frameNumer, frame=frame,
                dontWriteFrameIfExists=dontWriteFrameIfExists, verbose=verbose)

        # Detect mouths in frames
        if detectAndSaveMouths:
            face = detect_mouth_and_write(saveDir=saveDir,
                wordFileName=wordFileName, frameNumer=frameNumer, frame=frame,
                detector=detector, predictor=predictor,
                dontWriteMouthIfExists=dontWriteMouthIfExists, prevFace=face,
                verbose=verbose)

    return 0


def extract_frames_from_video(wordFileName, verbose=False):
    # Video file name
    videoFileName = '.'.join(wordFileName.split('.')[:-1]) + '.mp4'

    # Handle file not found
    if not os.path.isfile(videoFileName):
        raise ValueError("\n\nERROR: Video file not found:" + videoFileName + \
            "(extract_frames_from_video)\n\n")

    # Read video frames
    videoFrames = imageio.get_reader(videoFileName, 'ffmpeg')

    if verbose:
            print("Frames extracted from video:", videoFileName,
                "(extract_frames_from_video)")

    # Return
    return videoFrames


def read_jpeg_frames_from_dir(saveDir, wordFileName, verbose=False):
    
    # Frame names end with numbers from 00 to 30, so [0-3][0-9]
    videoFrameNamesFormat = os.path.join(saveDir,
                               "/".join(wordFileName.split("/")[-3:]).split('.')[0] + \
                               '_[0-3][0-9].jpg')

    # Read video frame names
    videoFrameNames = sorted(glob.glob(videoFrameNamesFormat))

    try:
        # Read all frame images
        videoFrames = []
        for frameName in videoFrameNames:
            videoFrames.append(imageio.imread(frameName))
    except OSError:
        # If not able to read
        raise ValueError("ERROR: could not read " + frameName + " (read_jpeg_frames_from_dir)")

    if verbose:
            print("Frames read from jpeg images:", wordFileName,
                "(read_jpeg_frames_from_dir)")

    # Return
    return videoFrames, videoFrameNames


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


def write_frame_image(saveDir, wordFileName, frameNumer, frame,
        dontWriteFrameIfExists=True, verbose=False):

    # Name
    frameImageName = os.path.join(saveDir, "/".join(wordFileName.split(
        "/")[-3:]).split('.')[0] + "_{0:02d}".format(frameNumer) + ".jpg")

    # If file is not supposed to be written if it exists
    if dontWriteFrameIfExists:
        # Check if file exists
        if os.path.isfile(frameImageName):
            if verbose:
                print("Frame image exists, so not written:", frameImageName,
                    "(write_frame_image)")
            # Return if file exists
            return

    # Write
    imageio.imwrite(frameImageName, frame)

    if verbose:
        print("Frame image written:", frameImageName, "(write_frame_image)")


def detect_mouth_and_write(saveDir, wordFileName, frameNumer, frame, detector, predictor,
        dontWriteMouthIfExists=True, prevFace=dlib.rectangle(30, 30, 220, 220),
        verbose=False):

    # Image Name
    mouthImageName = os.path.join(saveDir, "/".join(wordFileName.split(
                                  "/")[-3:]).split('.')[0] + \
                                  "_{0:02d}_mouth".format(frameNumer) + ".jpg")

    # If file is not supposed to be written if it exists
    if dontWriteMouthIfExists:
        # Check if file exists
        if os.path.isfile(mouthImageName):
            if verbose:
                print("Mouth image", mouthImageName,
                    "exists, so not detected. (detect_mouth_and_write)")
            # Return if file exists
            return prevFace

    # Detect and save mouth in frame
    mouthImage, face = detect_mouth_in_frame(frame, detector, predictor,
                                             prevFace=prevFace, verbose=verbose)

    # Save mouth image
    imageio.imwrite(mouthImageName, mouthImage)

    if verbose:
        print("Mouth image written:", mouthImageName, "(detect_mouth_and_write)")

    # Return
    return face


def detect_mouth_in_frame(frame, detector, predictor,
                          prevFace=dlib.rectangle(30, 30, 220, 220),
                          verbose=False):
    # Shape Coords: ------> x (cols)
    #               |
    #               |
    #               v
    #               y
    #             (rows)

    # Detect all faces
    faces = detector(frame, 1)

    # If no faces are detected
    if len(faces) == 0:
        if verbose:
            print("No faces detected, using prevFace", prevFace, "(detect_mouth_in_frame)")
        faces = [prevFace]

    # If multiple faces in frame, find the correct face by checking mouth mean
    if len(faces) > 1:

        # Iterate over the faces
        for face in faces:

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

            # Check if correct face is selected by checking position of mouth mean
            mouthMean = np.mean(mouthCoords, axis=0)
            if mouthMean[0] > 110 and mouthMean[0] < 150 \
                    and mouthMean[1] > 140 and mouthMean[1] < 170:
                break

    # If only one face in frame
    else:
        # Note face
        face = faces[0]
        # Predict facial landmarks
        shape = predictor(frame, face)
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
    expandedMouthRect = expand_rect(mouthRect,
        scale=(MOUTH_TO_FACE_RATIO * face.width() / mouthRect[2]),
        frame_shape=(frame.shape[0], frame.shape[1]))

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


def expand_rect(rect, scale=1.5, frame_shape=(256, 256)):
    # Rect: (x, y, w, h)
    w = int(rect[2] * scale)
    h = int(rect[3] * scale)
    x = max(0, min(frame_shape[1] - w, rect[0] - int((w - rect[2]) / 2)))
    y = max(0, min(frame_shape[0] - h, rect[1] - int((h - rect[3]) / 2)))
    return (x, y, w, h)


# #############################################################
# # REPROCESS IMAGES WITH MULTIPLE FACES
# # Find mouth means
# #############################################################


def reprocess_videos_with_multiple_faces(startExtracting=False,
        startSetWordNumber='test/AFTERNOON_00001',
        endSetWordNumber='test/ALWAYS_00001'):
    # Init
    dataDir = LRW_DATA_DIR
    saveDir = LRW_SAVE_DIR
    detector, predictor = load_detector_and_predictor()

    # LOOP
    for wordDir in tqdm.tqdm(sorted(glob.glob(os.path.join(dataDir, '*/')))):
        print(wordDir)

        for setDir in tqdm.tqdm(sorted(glob.glob(os.path.join(wordDir, '*/')))):
            print(setDir)
            wordFileNames = sorted(glob.glob(os.path.join(setDir, '*.txt')))

            for wordFileName in tqdm.tqdm(wordFileNames):

                if startSetWordNumber in wordFileName:
                    startExtracting = True
                if not startExtracting:
                    continue
                if endSetWordNumber is not None:
                    if endSetWordNumber in wordFileName:
                        return

                print(wordFileName)

                # Check if there are multiple faces in any of the frames of this video
                reprocess = False
                for i in range(1, 31):
                    frameName = os.path.join(saveDir, "/".join(wordFileName.split("/")[-3:]).split('.')[0] + '_{0:02d}.jpg'.format(i))
                    frame = imageio.imread(frameName)
                    faces = detector(frame, 1)
                    # If there are multiple faces
                    if len(faces) > 1:
                        # Set for reprocessing
                        reprocess = True
                        break

                # Reprocess
                if reprocess:

                    # Find setWordNumber for this and next video
                    oneStartSetWordNumber = '/'.join(wordFileName.split('/')[-2:]).split('.')[-2]
                    oneEndSetWordNumber = oneStartSetWordNumber.split('_')[0] + "_{0:05d}".format(int(oneStartSetWordNumber.split('_')[1]) + 1)

                    # If last video in folder
                    if not os.path.isfile(os.path.join('/'.join(wordFileName.split('/')[:-2]), oneEndSetWordNumber) + '.txt'):
                        sets = ['test', 'train', 'val']
                        newSet = sets[(sets.index(oneEndSetWordNumber.split('/')[0]) + 1) % 3]
                        if newSet == 'test':
                            newWord = LRW_VOCAB[LRW_VOCAB.index(oneEndSetWordNumber.split('/')[1].split('_')[0]) + 1]
                        else:
                            newWord = oneEndSetWordNumber.split('/')[1].split('_')[0]
                        oneEndSetWordNumber = newSet + '/' + newWord + '_00001'

                    print("\n", len(faces), "found in", frameName, "\n", oneStartSetWordNumber, oneEndSetWordNumber, "\n")
                    process_lrw(dataDir=LRW_DATA_DIR,
                        saveDir=LRW_SAVE_DIR,
                        startExtracting=False,
                        startSetWordNumber=oneStartSetWordNumber,
                        endSetWordNumber=oneEndSetWordNumber,
                        copyTxtFile=False,
                        extractAudioFromMp4=False,
                        dontWriteAudioIfExists=False,
                        extractFramesFromMp4=False,
                        writeFrameImages=False,
                        dontWriteFrameIfExists=True,
                        detectAndSaveMouths=True,
                        dontWriteMouthIfExists=False,
                        verbose=True)

# shape = predictor(frame, face)
# mouthCoords = np.array([[shape.part(i).x, shape.part(i).y]
#     for i in range(MOUTH_SHAPE_FROM, MOUTH_SHAPE_TO)])
# myMean.append(np.mean(mouthCoords, axis=0))

# win = dlib.image_window()
# for i in range(1, 31):
#     frameName = os.path.join(saveDir, "/".join(wordFileName.split("/")[-3:]).split('.')[0] + '_{0:02d}.jpg'.format(i))
#     frame = imageio.imread(frameName)
#     faces = detector(frame, 1)
#     face0 = faces[0]
#     shape = predictor(frame, face0)
#     win.clear_overlay()
#     win.set_image(frame)
#     win.add_overlay(face0)
#     win.add_overlay(shape)
#     time.sleep(.5)



#############################################################
# PLOT WORD DURATIONS
#############################################################


def plot_frames_per_sample():

    # From assessor_functions
    # lrw_n_of_frames_per_sample = load_array_of_var_per_sample_from_csv(csv_file_name=N_OF_FRAMES_PER_SAMPLE_CSV_FILE, collect_type=collect_type, collect_by='sample')

    plt.bar(np.arange(22), np.bincount(lrw_n_of_frames_per_sample))
    plt.xticks(range(22))

    plt.title('Histogram of number of frames per sample in LRW')
    plt.show()

    plt.figure()

    plt.subplot(131)
    plt.bar(np.arange(22), np.bincount(lrw_n_of_frames_per_sample_test))
    plt.xticks(range(22))
    plt.xlabel('Number of frames in sample')
    plt.ylabel('Number of sample')
    plt.title('Histogram of number of frames per sample in LRW test')

    plt.subplot(132)
    plt.bar(np.arange(22), np.bincount(lrw_n_of_frames_per_sample_train))
    plt.xticks(range(22))
    plt.xlabel('Number of frames in sample')
    plt.ylabel('Number of sample')
    plt.title('Histogram of number of frames per sample in LRW train')

    plt.subplot(133)
    plt.bar(np.arange(22), np.bincount(lrw_n_of_frames_per_sample_val))
    plt.xticks(range(22))
    plt.xlabel('Number of frames in sample')
    plt.ylabel('Number of sample')
    plt.title('Histogram of number of frames per sample in LRW val')

    plt.show()


def plot_word_duration_histograms(dataDir=LRW_DATA_DIR):
    lrw_train_number_of_frames, lrw_val_number_of_frames, lrw_test_number_of_frames = extract_all_word_number_of_frames(dataDir)
    # Train
    plt.figure()
    plt.subplot(131)
    a = plt.hist(lrw_train_number_of_frames, bins=np.arange(min(lrw_train_number_of_frames), max(lrw_train_number_of_frames)+1), align='left', rwidth=0.8)
    plt.xticks(range(max(a[1])))
    plt.xlabel('Number of frames in word'); plt.ylabel('Number of words'); plt.title('Histogram of number of frames per word in LRW train')
    # Val
    plt.subplot(132)
    a = plt.hist(lrw_val_number_of_frames, bins=np.arange(min(lrw_val_number_of_frames), max(lrw_val_number_of_frames)+1), align='left', rwidth=0.8)
    plt.xticks(range(max(a[1])))
    plt.xlabel('Number of frames in word'); plt.ylabel('Number of words'); plt.title('Histogram of number of frames per word in LRW val')
    # Test
    plt.subplot(133)
    a = plt.hist(lrw_test_number_of_frames, bins=np.arange(min(lrw_test_number_of_frames), max(lrw_test_number_of_frames)+1), align='left', rwidth=0.8)
    plt.xticks(range(max(a[1])))
    plt.xlabel('Number of frames in word'); plt.ylabel('Number of words'); plt.title('Histogram of number of frames per word in LRW test')
    # ALL
    plt.figure()
    a = plt.hist(np.append(np.append(lrw_train_number_of_frames, lrw_val_number_of_frames), lrw_test_number_of_frames),
        bins=np.arange(min(min(lrw_train_number_of_frames), min(lrw_val_number_of_frames), min(lrw_test_number_of_frames)),
            max(max(lrw_train_number_of_frames), max(lrw_val_number_of_frames), max(lrw_test_number_of_frames))+1),
        align='left', rwidth=0.8)
    plt.xticks(range(max(a[1])))
    plt.xlabel('Number of frames in word'); plt.ylabel('Number of words'); plt.title('Histogram of number of frames per word in LRW')
    plt.show()


def extract_all_word_number_of_frames(dataDir=LRW_DATA_DIR):
    lrw_train_number_of_frames = []
    lrw_val_number_of_frames = []
    lrw_test_number_of_frames = []
    for wordDir in tqdm.tqdm(sorted(glob.glob(os.path.join(dataDir, '*/')))):
        for setDir in tqdm.tqdm(sorted(glob.glob(os.path.join(wordDir, '*/')))):
            wordFileNames = sorted(glob.glob(os.path.join(setDir, '*.txt')))
            for wordFileName in tqdm.tqdm(wordFileNames):
                    line = read_last_line_in_file(wordFileName)
                    if 'train' in wordFileName:
                         lrw_train_number_of_frames.append(int(float(line.rstrip().split()[-2])*VIDEO_FPS))
                    if 'val' in wordFileName:
                         lrw_val_number_of_frames.append(int(float(line.rstrip().split()[-2])*VIDEO_FPS))
                    if 'test' in wordFileName:
                         lrw_test_number_of_frames.append(int(float(line.rstrip().split()[-2])*VIDEO_FPS))
    return lrw_train_number_of_frames, lrw_val_number_of_frames, lrw_test_number_of_frames


def read_last_line_in_file(wordFileName):
    try:
        with open(wordFileName) as f:
            for line in f:
                 pass
        return line
    except OSError:
        read_last_line_in_file(wordFileName)


#############################################################
# PLOT WORD DURATIONS
#############################################################


def plot_word_duration_histograms(dataDir=LRW_DATA_DIR):
    lrw_train_number_of_frames, lrw_val_number_of_frames, lrw_test_number_of_frames = extract_all_word_number_of_frames(dataDir)
    # Train
    plt.figure()
    plt.subplot(131)
    a = plt.hist(lrw_train_number_of_frames, bins=np.arange(min(lrw_train_number_of_frames), max(lrw_train_number_of_frames)+1), align='left', rwidth=0.8)
    plt.xticks(range(max(a[1])))
    plt.xlabel('Number of frames in word'); plt.ylabel('Number of words'); plt.title('Histogram of number of frames per word in LRW train')
    # Val
    plt.subplot(132)
    a = plt.hist(lrw_val_number_of_frames, bins=np.arange(min(lrw_val_number_of_frames), max(lrw_val_number_of_frames)+1), align='left', rwidth=0.8)
    plt.xticks(range(max(a[1])))
    plt.xlabel('Number of frames in word'); plt.ylabel('Number of words'); plt.title('Histogram of number of frames per word in LRW val')
    # Test
    plt.subplot(133)
    a = plt.hist(lrw_test_number_of_frames, bins=np.arange(min(lrw_test_number_of_frames), max(lrw_test_number_of_frames)+1), align='left', rwidth=0.8)
    plt.xticks(range(max(a[1])))
    plt.xlabel('Number of frames in word'); plt.ylabel('Number of words'); plt.title('Histogram of number of frames per word in LRW test')
    # ALL
    plt.figure()
    a = plt.hist(np.append(np.append(lrw_train_number_of_frames, lrw_val_number_of_frames), lrw_test_number_of_frames),
        bins=np.arange(min(min(lrw_train_number_of_frames), min(lrw_val_number_of_frames), min(lrw_test_number_of_frames)),
            max(max(lrw_train_number_of_frames), max(lrw_val_number_of_frames), max(lrw_test_number_of_frames))+1),
        align='left', rwidth=0.8)
    plt.xticks(range(max(a[1])))
    plt.xlabel('Number of frames in word'); plt.ylabel('Number of words'); plt.title('Histogram of number of frames per word in LRW')
    plt.show()


def extract_all_word_number_of_frames(dataDir=LRW_DATA_DIR):
    lrw_train_number_of_frames = []
    lrw_val_number_of_frames = []
    lrw_test_number_of_frames = []
    for wordDir in tqdm.tqdm(sorted(glob.glob(os.path.join(dataDir, '*/')))):
        for setDir in tqdm.tqdm(sorted(glob.glob(os.path.join(wordDir, '*/')))):
            wordFileNames = sorted(glob.glob(os.path.join(setDir, '*.txt')))
            for wordFileName in tqdm.tqdm(wordFileNames):
                    line = read_last_line_in_file(wordFileName)
                    if 'train' in wordFileName:
                         lrw_train_number_of_frames.append(int(float(line.rstrip().split()[-2])*VIDEO_FPS))
                    if 'val' in wordFileName:
                         lrw_val_number_of_frames.append(int(float(line.rstrip().split()[-2])*VIDEO_FPS))
                    if 'test' in wordFileName:
                         lrw_test_number_of_frames.append(int(float(line.rstrip().split()[-2])*VIDEO_FPS))
    return lrw_train_number_of_frames, lrw_val_number_of_frames, lrw_test_number_of_frames


def read_last_line_in_file(wordFileName):
    try:
        with open(wordFileName) as f:
            for line in f:
                 pass
        return line
    except OSError:
        read_last_line_in_file(wordFileName)


#############################################################
# RUN ON ONE IMAGE
#############################################################


def test_mouth_detection_in_frame(dataDir=LRW_SAVE_DIR, saveDir=LRW_SAVE_DIR,
                                  word="ABOUT", set="train", number=1, frameNumber=1,
                                  scaleFactor=MOUTH_TO_FACE_RATIO, showMouthOnFrame=True,
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
        saveDir, word, set, word + '_{0:05d}'.format(number) + '.txt')

    frame = read_jpeg_frames_from_dir(saveDir, wordFileName)[frameNumber]
    try:
        face = detector(frame, 1)[0]
    except IndexError:
        face = dlib.rectangle(30, 30, 220, 220)
    
    shape = predictor(frame, face)

    # # Show landmarks and face
    # win = dlib.image_window()
    # win.set_image(frame)
    # win.add_overlay(face)
    # win.add_overlay(shape)

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

