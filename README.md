# lipreading-in-the-wild-experiments

This repository contains my experiments with lip reading using deep learning in Keras. I am train and test on the [LRW dataset](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/).

## Requirements

1. Required libraries by Python (I used Python3) are mentioned in the requirements.txt file.

2. 'ffmpeg' is also required for the imageio module to read an mp4 file. Install it using:

```sh
sudo apt install ffmpeg
```

3. "shape_predictor_68_face_landmarks.dat" file is required inside the _shape-predictor_ directory. It is used for detecting facial landmarks. Instructions to download and place the file are available in the readme file in the _shape-predictor_ directory.

## Files

The files contained are:

- process_lrw.py

	- Code to automatically process the lRW dataset: extract and save frames from all mp4 videos, detect and save mouths from all frames

	- There are options to extract/not extract frames from videos (if they are already extracted), detect/not detect mouth images, save/not save the images (frames or mouths)

## Process to detect and save mouths:

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
	- Expand to set the width of the mouth bounding box as 0.6 times the width of _face_

- Resize the mouth to 120x120 pixels
	- Mouth is the portion of frame given by thhe expanded mouth bounding box	
	- [SyncNet paper](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf) resizes mouth to 111x111, [LRW paper](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16/chung16.pdf) resizes to 112x112
	- I resize to 120x120 so that random cropping can be employed to extract 112x112-sized images

- Save the resized mouth image



