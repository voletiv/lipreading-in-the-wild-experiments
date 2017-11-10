# lipreading-in-the-wild-experiments

This repository contains my experiments with lip reading using deep learning in Keras. I train and test on the [LRW dataset](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/).

## process-lrw/

Codes to convert the videos in LRW Dataset to:
    - frames pertaining to words
    - mouth area of those frames
    - audio 

Instructions are provided in README file in directory.

## shape-predictor/

Directory to place the "_shape\_predictor\_68\_face\_landmarks.dat"_ file, required by _process-lrw_ and _head-pose_

## image-retrieval/

Codes and files --- considering the lipreader as an image retrieval system

## head-pose/

Codes and files --- to compute head pose in all frames in LRW dataset (extracted using _process-lrw_)

Head pose is determined using [voletiv/deepgaze](https://github.com/voletiv/deepgaze) (my fork of [deepgaze](https://github.com/mpatacchiola/deepgaze).

Instructions are provided in README file in directory.

