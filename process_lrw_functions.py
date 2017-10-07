from process_lrw_params import *

#############################################################
# EXTRACT AUDIO, FRAMES, AND MOUTHS
#############################################################


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
                    try:
                        copy_txt_file(wordFileName, verbose)
                    except ValueError as err:
                        print(err)
                    except KeyboardInterrupt:
                        print("\n\nCtrl+C was pressed!\n\n")
                        return

                # Extract audio
                if extractAudioFromMp4:
                    try:
                        extract_audio_from_mp4(wordFileName=wordFileName,
                            dontWriteAudioIfExists=dontWriteAudioIfExists, verbose=verbose)
                    except ValueError as err:
                        print(err)
                    except KeyboardInterrupt:
                        print("\n\nCtrl+C was pressed!\n\n")
                        return

                # If frame doesn't need ot be extracted from mp4 video,
                # or mouth doesn't need to be detected, continue
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
                        print("\n\nCtrl+C was pressed!\n\n")
                        return 1

                # Extracting
                extractReturn = please_extract(videoFile)

                # If Ctrl+C was pressed
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
            print("Detector and Predictor loaded. (load_detector_and_predictor)")
        return detector, predictor
    # If error in SHAPE_PREDICTOR_PATH
    except RuntimeError:
        raise ValueError("ERROR: Wrong Shape Predictor .dat file path - " + \
            SHAPE_PREDICTOR_PATH, "(load_detector_and_predictor)")


def copy_txt_file(wordFileName, verbose=False):
    # Names
    fromFileName = wordFileName
    toFileName = os.path.join(LRW_SAVE_DIR, "/".join(wordFileName.split("/")[-3:]))
    try:
        shutil.copyfile(fromFileName, toFileName)
        if verbose:
            print("Text file copied:", fromFileName, "->", toFileName,
                "(copy_txt_file)")
        return 0
    except:
        raise ValueError("ERROR: shutil failed to copy " + fromFileName + \
            " to " + toFileName + " (copy_txt_file)")


def extract_audio_from_mp4(wordFileName, dontWriteAudioIfExists, verbose=False):
    # Names
    videoFileName = '.'.join(wordFileName.split('.')[:-1]) + '.mp4'
    audioFileName = os.path.join(LRW_SAVE_DIR,
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
        raise ValueError("ERROR: Audio file " + audioFileName + " NOT WRITEN!! (extract_audio_from_mp4)")

    if verbose:
        if commandReturn == 0:
            print("Audio file written:", audioFileName, "(extract_audio_from_mp4)")


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
            print("Frames extracted from video:", videoFileName,
                "(extract_frames_from_video)")
    # Return
    return videoFrames


def read_jpeg_frames_from_dir(wordFileName, verbose=False):
    
    # Frame names end with numbers from 00 to 30, so [0-3][0-9]
    videoFrameNames = sorted(
        glob.glob(os.path.join(LRW_SAVE_DIR,
                               "/".join(wordFileName.split("/")[-3:]).split('.')[0] + \
                               '_[0-3][0-9].jpg')))

    videoFrames = []
    # Read all frame images
    for frameName in videoFrameNames:
        videoFrames.append(imageio.imread(frameName))

    if verbose:
            print("Frames read from jpeg images:", wordFileName, "(read_jpeg_frames_from_dir)")

    # Return
    return videoFrames


def write_frame_image(wordFileName, f, frame, dontWriteFrameIfExists=True,
        verbose=False):

    # Name
    frameImageName = os.path.join(LRW_SAVE_DIR, "/".join(wordFileName.split(
        "/")[-3:]).split('.')[0] + "_{0:02d}".format(f + 1) + ".jpg")

    # If file is not supposed to be written if it exists
    if dontWriteFrameIfExists:
        # Check if file exists
        if os.path.isfile(frameImageName):
            if verbose:
                print("Frame image exists, so not written:", frameImageName, "(write_frame_image)")
            # Return if file exists
            return

    # Write
    imageio.imwrite(frameImageName, frame)

    if verbose:
        print("Frame image written:", frameImageName, "(write_frame_image)")


def detect_mouth_and_write(wordFileName, f, frame, detector, predictor,
        dontWriteMouthIfExists=True, prevFace=dlib.rectangle(30, 30, 220, 220),
        verbose=False):

    # Image Name
    mouthImageName = os.path.join(LRW_SAVE_DIR, "/".join(wordFileName.split(
                                  "/")[-3:]).split('.')[0] + \
                                  "_{0:02d}_mouth".format(f + 1) + ".jpg")

    # If file is not supposed to be written if it exists
    if dontWriteMouthIfExists:
        # Check if file exists
        if os.path.isfile(mouthImageName):
            if verbose:
                print("Mouth image", mouthImageName, "exists, so not detected. (detect_mouth_and_write)")
            # Return if file exists
            return face

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

