## process-lrw/

Codes to convert the videos in LRW Dataset to:

    - frames pertaining to words

    - (aligned or unaligned) mouth area of those frames

    - audio 

Please see `process_lrw.py` for the single function to do this.

### Requirements

1. Libraries required by Python (I used Python3) are mentioned in the _requirements.txt_ file in the parent directory.

2. '_ffmpeg_' is also required for the imageio module to read an mp4 file. Install it using:

```sh
sudo apt install ffmpeg
```

3. "_shape\_predictor\_68\_face\_landmarks.dat"_ file is required inside the _shape-predictor_ directory. It is used for detecting facial landmarks. Instructions to download and place the file are available in the readme file inside the _shape-predictor_ directory.

### Files

The files contained are:

- `process_lrw.py`

    - Code to automatically process the LRW dataset.

    - There are options to copy (or) not copy the .txt files containing metadata, extract (or) not extract the audio aac from the video, extract (or) not extract frames from video (if they are already extracted), detect (or) not detect mouth images, save (or) not save the images (frames or mouths):

        - dataDir: the directory with the dataset,

        - saveDir: to the directory the files need to be saved in,

        - startExtracting (bool): start (or) don't start extracting right away, i.e. wait until the "startSetWordNumber" is reached,

        - startSetWordNumber: the set {test, train, val}, Word {ABOUT, ...}, Number {0..50 for test, val; 0..1000 for train} to start processing from; for eg. 'test/ABOUT_00001'

        - endSetWordNumber: the set {test, train, val}, Word {ABOUT, ...}, Number {0..50 for test, val; 0..1000 for train} before which to end processing; for eg. 'val/ABOUT_00050'

        - copyTxtFile (bool): copy (or) don't copy the .txt files containing metadata,

        - extractAudioFromMp4 (bool): extract (or) don't extract audio,

        - dontWriteAudioIfExists (bool): don't write (or) write the audio file if (or) even if it has already been written,

        - extractFramesFromMp4 (bool): extract frames from mp4 videos, or read the frame images in saveDir,

        - writeFrameImages (bool): save the frames extracted,

        - dontWriteFrameIfExists (bool): don't write/write the frame images file if/even if they have already been written,

        - detectAndSaveMouths (bool): detect and save mouths from all frames,

        - dontWriteMouthIfExists (bool): don't write (or) write the mouth images file if (or) even if they have already been written,

        - verbose (bool): print debugging logs

- `process_lrw_functions.py`

    - All functions related to process_lrw

- `process_lrw_params.py`

    - All parameters related to process_lrw

### Process to detect and save mouths:

Each video _frame_ is a colour image (3 channels) 256x256 pixels in size.

- Detect _face_ using dlib detector
    - _face_ is a dlib.rectangle object of the rectangle identifying the face in the frame
    - If face is not detected, use the previously detected face
    - By default (if no face has ever been detected before), assume the face rectangle has (left, top, right, bottom) pixel coordinates as (20, 20, 230, 230) in the 256x256 frame.

- Detect _shape_, the facial landmarks in _face_ using the dlib Shape Predictor
    - dlib Shape Predictor uses "shape_predictor_68_face_landmarks.dat" file, to be placed in the shape-predictor directory (instructions available in the readme file in the directory)
    - _shape_ is a set of points on _frame_ corresponding to the different facial landmarks on _face_. Points 48 to 67 correspond to the mouth.

- Detect mouth as the bounding box of all the mouth points in _shape_

- Make the mouth bounding box square, with the side as the larger of width and height of the mouth bounding box

- Expand the mouth bounding box
    - Expand to set the width of the mouth bounding box as 0.65 times the width of _face_

- Resize the mouth to 120x120 pixels
    - Mouth is the portion of frame given by thhe expanded mouth bounding box   
    - [SyncNet paper](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf) resizes mouth to 111x111, [LRW paper](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16/chung16.pdf) resizes to 112x112
    - I resize to 120x120 so that random cropping can be employed to extract 112x112-sized images

- Save the resized mouth image

## shape-predictor/

Directory to place the "_shape\_predictor\_68\_face\_landmarks.dat"_ file, required by _process-lrw_ and _head-pose_

