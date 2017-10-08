from process_lrw_params import *

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
        print("\n\nERROR: dataDir is not a valid directory! Input dataDir:", dataDir, "\n\n")
        return

    if not os.path.isdir(saveDir):
        print("\n\nERROR: saveDir is not a valid directory! Input saveDir:", saveDir, "\n\n")
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

    # For each word
    for wordDir in tqdm.tqdm(sorted(glob.glob(os.path.join(dataDir, '*/')))):
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

                # If frames don't need ot be extracted from mp4 video,
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
            videoFrames = read_jpeg_frames_from_dir(saveDir, wordFileName, verbose)
    # If mp4 or jpeg files to read are missing
    except ValueError as err:
        raise ValueError(err)

    # Default face bounding box
    if detectAndSaveMouths:
        # Default face
        face = dlib.rectangle(30, 30, 220, 220)

    # For each frame
    for f, frame in enumerate(videoFrames):

        # Write the frame image (from video)
        if extractFramesFromMp4 and writeFrameImages:
            write_frame_image(saveDir=saveDir, wordFileName=wordFileName, f=f,
                frame=frame, dontWriteFrameIfExists=dontWriteFrameIfExists,
                verbose=verbose)

        # Detect mouths in frames
        if detectAndSaveMouths:
            face = detect_mouth_and_write(saveDir=saveDir, wordFileName=wordFileName,
                f=f, frame=frame, detector=detector, predictor=predictor,
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
    videoFrameNames = sorted(glob.glob(videoFrameNamesFormat))

    # If no frames are read
    if len(videoFrameNames) < 30:
        raise ValueError("\n\nERROR: 30 frames not found in" + \
            videoFrameNamesFormat + " format. (read_jpeg_frames_from_dir)\n\n")

    # Read all frame images
    videoFrames = []
    for frameName in videoFrameNames:
        videoFrames.append(imageio.imread(frameName))

    if verbose:
            print("Frames read from jpeg images:", wordFileName,
                "(read_jpeg_frames_from_dir)")

    # Return
    return videoFrames


def write_frame_image(saveDir, wordFileName, f, frame, dontWriteFrameIfExists=True,
        verbose=False):

    # Name
    frameImageName = os.path.join(saveDir, "/".join(wordFileName.split(
        "/")[-3:]).split('.')[0] + "_{0:02d}".format(f + 1) + ".jpg")

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


def detect_mouth_and_write(saveDir, wordFileName, f, frame, detector, predictor,
        dontWriteMouthIfExists=True, prevFace=dlib.rectangle(30, 30, 220, 220),
        verbose=False):

    # Image Name
    mouthImageName = os.path.join(saveDir, "/".join(wordFileName.split(
                                  "/")[-3:]).split('.')[0] + \
                                  "_{0:02d}_mouth".format(f + 1) + ".jpg")

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
                                             prevFace=prevFace)

    # Save mouth image
    imageio.imwrite(mouthImageName, mouthImage)

    if verbose:
        print("Mouth image", mouthImageName, "written. (detect_mouth_and_write)")

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

    # Detect all faces
    faces = detector(frame, 1)

    # If no faces are detected
    if len(faces) == 0:
        face = [prevFace]

    # Iterate over the faces, find the correct one by checking mouth mean
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

    # Mouth Rect: x, y, w, h
    mouthRect = (np.min(mouthCoords[:, 0]), np.min(mouthCoords[:, 1]),
                 np.max(mouthCoords[:, 0]) - np.min(mouthCoords[:, 0]),
                 np.max(mouthCoords[:, 1]) - np.min(mouthCoords[:, 1]))

    # Make mouthRect square
    mouthRect = make_rect_shape_square(mouthRect)

    # Expand mouthRect square
    expandedMouthRect = expand_rect(
        mouthRect, scale=(MOUTH_TO_FACE_RATIO * face.width() / mouthRect[2]))

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


# #############################################################
# # FIND MOUTH MEANS, AND IMAGES WITH MULTIPLE FACES
# #############################################################

# # extract_and_save_audio_frames_and_mouths_from_dir
# dataDir = LRW_DATA_DIR
# saveDir = LRW_SAVE_DIR
# detector, predictor = load_detector_and_predictor()

# myMean = []

# startSetWordNumber='train/ACROSS_00447'
# startExtracting=False
# # For each word
# for wordDir in tqdm.tqdm(sorted(glob.glob(os.path.join(dataDir, '*/')))):
#     print(wordDir)
#     # train, val or test
#     for setDir in tqdm.tqdm(sorted(glob.glob(os.path.join(wordDir, '*/')))):
#         print(setDir)
#         # Read all .txt file names (since .txt are saved in both
#         # LRW_DATA_DIR and LRW_SAVE_DIR)
#         wordFileNames = sorted(glob.glob(os.path.join(setDir, '*.txt')))
#         # For each video
#         for wordFileName in tqdm.tqdm(wordFileNames):
#             if startSetWordNumber in wordFileName:
#                 startExtracting = True
#             if startExtracting:
#                 print(wordFileName)
#                 frameName = os.path.join(saveDir, "/".join(wordFileName.split("/")[-3:]).split('.')[0] + '_01.jpg')
#                 frame = imageio.imread(frameName)
#                 faces = detector(frame, 1)

#                 shape = predictor(frame, face)
#                 mouthCoords = np.array([[shape.part(i).x, shape.part(i).y]
#                     for i in range(MOUTH_SHAPE_FROM, MOUTH_SHAPE_TO)])
#                 myMean.append(np.mean(mouthCoords, axis=0))

#                 print(len(faces))
#                 if len(faces) > 1:
#                     raise KeyboardInterrupt
#                 del faces

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

