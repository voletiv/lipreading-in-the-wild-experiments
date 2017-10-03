import cv2
import glob
# stackoverflow.com/questions/29718238/how-to-read-mp4-video-to-be-processed-by-scikit-image
import imageio
import matplotlib
# matplotlib.use('agg')     # Use this for remote terminals
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm

from matplotlib.patches import Rectangle
import dlib

# Facial landmark detection
# http://dlib.net/face_landmark_detection.py.html

#############################################################
# PARAMS
#############################################################

LRW_DATA_DIR = '/media/voletiv/01D2BF774AC76280/Datasets/LRW/lipread_mp4/'
LRW_SAVE_DIR = '/home/voletiv/Datasets/LRW/lipread_mp4'

videoFile = 'media/voletiv/01D2BF774AC76280/Datasets/LRW/lipread_mp4/ABOUT/test/ABOUT_00001.mp4'
frameFile = '/home/voletiv/Datasets/LRW/lipread_mp4/ABOUT/test/ABOUT_00001_01.jpg'


#############################################################
# DETECT AND SAVE MOUTH REGIONS
#############################################################

detector = dlib.get_frontal_face_detector()


# Convert dlib rectangle to bounding box (x, y, w, h)
def rect_to_bb(rect):
    if isinstance(rect, dlib.rectangle):
        # take a bounding predicted by dlib and convert it
        # to the format (x, y, w, h) as we would normally do
        # with OpenCV
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
    else:
        x, y, w, h = rect
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


# Find rectangle bounding face
def findFaceRect(frame, mode='dlib'):
    # Detect face using dlib detector
    faceRect = detector(frame, 1)
    # If at least 1 face is found
    if len(faceRect) > 0:
        return faceRect[0]
    # If no face is found
    else:
        return ()


# Gaussian kernel
# https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# Find mean pixel value of mouth area in expanded face, assuming faceW = 128
def findMouthMeanInFaceRect(face, wReduceFactor=0.6, wLeftOffsetFactor=0.0, hTopReduceFactor=0.5, hBottomOffsetFactor=0.15, showACh=False, aChThresh=0.9):
    # Reduce frame width to find mouth in constrained area
    (faceH, faceW, _) = face.shape
    # plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)); plt.show()
    wDelta = wReduceFactor * faceW
    newFaceW = faceW - int(wDelta) + int(wLeftOffsetFactor * faceW)
    newFaceX = int(wDelta / 2)
    hDelta = hTopReduceFactor * faceH
    newFaceY = int(hTopReduceFactor * faceH)
    newFaceH = faceH - int(hDelta) - int(hBottomOffsetFactor * faceH)
    # Extract smaller face
    smallFace = np.array(
        face[newFaceY:newFaceY + newFaceH, newFaceX:newFaceX + newFaceW, :])
    # plt.imshow(cv2.cvtColor(smallFace, cv2.COLOR_BGR2RGB)); plt.show()
    # Convert face to LAB, extract A channel
    aCh = cv2.cvtColor(smallFace, cv2.COLOR_BGR2Lab)[:, :, 1]
    (aChW, aChH) = aCh.shape
    # Element-wise multiply with gaussian kernel with center pixel at
    # 30% height, and sigma 500
    gaussKernel = matlab_style_gauss2D(
        (aChW, 2 * 0.7 * aChH), sigma=500)[:, -aChH:]
    aCh = np.multiply(aCh, gaussKernel)
    # Rescale to [0, 1]
    aCh = (aCh - aCh.min()) / (aCh.max() - aCh.min())
    # Find mean of those pixels > 0.9
    # plt.imshow(aCh > 0.9, cmap='gray'); plt.show()
    if showACh:
        plt.imshow(aCh, cmap='gray')
        plt.show()
    # Here, the X & Y axes of np array are the Y & X of Rectangle respectively
    mouthY, mouthX = np.where(aCh > aChThresh)
    mouthXMean = mouthX.mean()
    mouthYMean = mouthY.mean()
    # plt.imshow(cv2.cvtColor(smallFace, cv2.COLOR_BGR2RGB)); ca = plt.gca(); ca.add_patch(Rectangle((mouthXMean - 2, mouthYMean - 2), 4, 4, edgecolor='g', facecolor='g')); plt.show()
    # plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)); ca = plt.gca(); ca.add_patch(Rectangle((newFaceX + mouthXMean - 2, newFaceY + mouthYMean - 2), 4, 4, edgecolor='g', facecolor='g')); plt.show()
    return (newFaceX + mouthXMean, newFaceY + mouthYMean, aCh)


# Extract mouth image
def extractMouthImage(frame, showACh=False, aChThresh=0.9, mode='dlib', mouthW=112):
    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); plt.show()
    # Find rectangle bounding face
    faceRect = findFaceRect(frame, mode=mode)
    if faceRect != ():
        faceX, faceY, faceW, faceH = rect_to_bb(faceRect)
    # If a face is not found, return empty
    else:
        return faceRect
    # plt.imshow(frame); ca = plt.gca(); ca.add_patch(Rectangle((faceX, faceY), faceW, faceH, edgecolor='r', fill=False)); plt.show()
    # plt.imshow(frame[faceY:faceY + faceH, faceX:faceX + faceW]); plt.show()
    # Return just the face
    face = frame[faceY:faceY + faceH, faceX:faceX + faceW]
    # plt.imshow(face); plt.show()
    # Find mean pixel value of mouth area - pixel value is
    # plottable on face, not frame
    mouthXMean, mouthYMean, aCh = findMouthMeanInFaceRect(
        face, wReduceFactor=0.6, hTopReduceFactor=0.5, hBottomOffsetFactor=0.1, showACh=showACh, aChThresh=aChThresh)
    # plt.imshow(face); ca = plt.gca(); ca.add_patch(Rectangle((mouthXMean-2, mouthYMean-2), 4, 4, edgecolor='g', facecolor='g')); plt.show()
    # plt.imshow(aCh, cmap='gray'); plt.show()
    # To make mouth mean plottable on frame
    frameMouthYMean = mouthYMean + faceY
    frameMouthXMean = mouthXMean + faceX
    # plt.imshow(frame); ca = plt.gca(); ca.add_patch(Rectangle((frameMouthXMean-2, frameMouthYMean-2), 4, 4, edgecolor='g', facecolor='g')); plt.show()
    # In case mouthYMean cannot cover 56 pixels of mouth around it
    if (int(frameMouthYMean + mouthW / 2) > frame.shape[0]):
        frameMouthYMean = frame.shape[0] - mouthW / 2
    # plt.imshow(face); ca = plt.gca(); ca.add_patch(Rectangle((mouthXMean-2, mouthYMean-2), 4, 4, edgecolor='g', facecolor='g')); plt.show()
    # Extract mouth as a colour 112x112 region around the mean
    mouth = frame[int(frameMouthYMean - mouthW / 2):int(frameMouthYMean + mouthW / 2),
                  int(frameMouthXMean - mouthW / 2):int(frameMouthXMean + mouthW / 2), :]
    # plt.imshow(mouth, cmap='gray'); plt.show()
    # # Minimize contrast
    # mouth = (mouth - mouth.min()) / (mouth.max() - mouth.min())
    return mouth, face, aCh


def extract_frames_and_mouths(videoFile):
    video = imageio.get_reader(videoFile, 'ffmpeg')
    # For each frame
    for f, frame in enumerate(video):
        # Not the name of file to be saved
        frame_name = os.path.join(LRW_SAVE_DIR, "/".join(videoFile.split(
            "/")[-3:]).split('.')[0] + "_{0:02d}".format(f + 1) + ".jpg")
        # Save if file doesn't exist
        if not os.path.isfile(frame_name):
            imageio.imwrite(frame_name, frame)
        # Convert Image array to np array
        frame = np.array(frame)
        # Extract the mouth
        mouth, face, aCh = extractMouthImage(
            frame, showACh=showACh, aChThresh=aChThresh, mode=mode, mouthW=mouthW)
        # plt.imshow(mouth, cmap='gray'); plt.show()
        # plt.imshow(face); plt.show()
        # plt.imshow(aCh, cmap='gray'); plt.show()
        # If a mouth has been found
        if mouth != ():
            # Write the name
            mouth_name = os.path.join(LRW_SAVE_DIR, "/".join(videoFile.split("/")[-3:]).split(
                '.')[0] + "_{0:02d}_mouth".format(f + 1) + ".jpg")
            # print("Saving mouth to:", mouth_name)
            # imwrite
            isWritten = imageio.imwrite(mouth_name, mouth)
    return 1

# Extract And Save Mouth Images
extract, startDir, showACh, aChThresh, mode, mouthW = False, 's25', False, 0.9, 'dlib', 112


def extractAndSaveMouthImages(rootDir=LRW_DATA_DIR,
                              extract=False,
                              startDir='train/ABOUT_00035',
                              showACh=False,
                              aChThresh=0.9,
                              mode='dlib',
                              mouthW=112):
    # For each word
    for wordDir in tqdm.tqdm(sorted(glob.glob(os.path.join(rootDir, '*/')))):
        print(wordDir)
        # train, val or test
        for setDir in tqdm.tqdm(sorted(glob.glob(os.path.join(wordDir, '*/')))):
            print(setDir)
            wordVids = sorted(glob.glob(os.path.join(setDir, '*.mp4')))
            # For each video
            for videoFile in tqdm.tqdm(wordVids):
                # Don't extract until all previously extract_return are passed
                if startDir in videoFile:
                    extract = True
                if extract:
                    # Extract
                    def extract_please(videoFile):
                        try:
                            extract_return = extract_frames_and_mouths(
                                videoFile)
                            return extract_return
                        except OSError:
                            print("Trying again...")
                            extract_return = extract_please(videoFile)
                            return extract_return
                        except KeyboardInterrupt:
                            print("Ctrl+C was pressed!")
                            return -1
                    extract_return = extract_please(videoFile)
                    if extract_return == -1:
                        return


#############################################################
# CONVERT VIDEOS TO IMAGES
# SAVE IMAGES IN INDIVIDUAL FOLDERS BELONGING TO EACH VIDEO
#############################################################


# Extract all video into frames
def extract_frames_and_mouths_from_all_videos(rootDir=LRW_DATA_DIR,
                                              extract=False,
                                              startDir='val/ACROSS_00001',
                                              extract_frame=True,
                                              extract_mouth=True):
    # For each word in LRW
    for wordDir in tqdm.tqdm(sorted(glob.glob(os.path.join(rootDir, '*/')))):
        print(wordDir)
        # train, val or test
        for setDir in tqdm.tqdm(sorted(glob.glob(os.path.join(wordDir, '*/')))):
            print(setDir)
            wordVids = sorted(glob.glob(os.path.join(setDir, '*.mp4')))
            # For each video
            for videoFile in tqdm.tqdm(wordVids):
                # Don't extract until all previously extract_return are passed
                if startDir in videoFile:
                    extract = True
                # Extract
                if extract:
                    # Handling OSError
                    def extract_please(videoFile):
                        try:
                            if extract_frame and not extract_mouth:
                                extract_return = extract_frames_and_save(
                                    videoFile, write=True)
                            elif extract_frame and extract_mouth:
                            return extract_return
                        except OSError:
                            print("Trying again...")
                            extract_return = extract_please(videoFile)
                            return extract_return
                        except KeyboardInterrupt:
                            print("Ctrl+C was pressed!")
                            return -1
                    # Extracting
                    extract_return = extract_please(videoFile)
                    if extract_return == -1:
                        return


def extract_and_save_frames_and_mouths(videoFile):
    # Capture the video
    video = extract_frames_and_save(videoFile, write=False)


# Extract video into frames
def extract_frames_and_mouths_and_save(videoFile, extract_mouth=True, write=False):
    # Capture the video
    video = imageio.get_reader(videoFile, 'ffmpeg')
    # Iterate over the frames
    for f, frame in enumerate(video):
        # Write frame
        if write:
            frame_name = os.path.join(LRW_SAVE_DIR, "/".join(videoFile.split(
                "/")[-3:]).split('.')[0] + "_{0:02d}".format(f + 1) + ".jpg")
            # If it doesn't exist already
            if not os.path.isfile(frame_name):
                imageio.imwrite(frame_name, frame)
        # Extract mouth
        if extract_mouth:

    # Return
    return video

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Extract mouth from frame
def extract_mouth_from_frame(frame):
    # Detect faces
    dets = detector(frame, 1)
    shape = predictor(img, d)


    #############################################################
    # PROCESS
    #############################################################

    # Copy directory structure from LRW_DATA_DIR to LRW_SAVE_DIR (in bash)
    # cd $LRW_DATA_DIR && find . type -d -exec mkdir -p -- $LRW_SAVE_DIR{} \;

    # Copy the txt files from LRW_DATA_DIR to LRW_SAVE_DIR (in bash)
    # cd $LRW_DATA_DIR && find . -name \*.txt -exec cp -parents {}
    # $LRW_SAVE_DIR \;

    # Extract frames
