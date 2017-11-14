import glob
import os
import sys

from head_pose_params import *

sys.path.append(DEEPGAZE_EXAMPLES_DIR)

from ex_cnn_head_pose_estimation_images_list import *

startWord = None
endWord = None

for i, v in enumerate(sys.argv):

    if "--startWord" in v or "-sW" in v:
        try:
            startWord = sys.argv[i+1]
        except IndexError:
            print("[ERROR] Please specify where to start from!")
            sys.exit()

    if "--endWord" in v or "-eW" in v:
        try:
            endWord = sys.argv[i+1]
        except IndexError:
            print("[ERROR] Please specify where to end at!")
            sys.exit()

run = False

for file in sorted(glob.glob(os.path.join(LRW_SAVE_DIR, 'head_pose_jpg_file_names*'))):
    # Start
    if startWord is not None:
        if startWord in file:
            run = True
    else:
        run = True
    # End
    if endWord is not None:
        if endWord in file:
            sys.exit()
    # Run
    if run:
        word = file.split('/')[-1].split('.')[-2].split('_')[-1]
        print(word)
        return_val = cnn_head_pose_estimation_images_list(file, save_npy=True, npy_file_name=os.path.join(LRW_SAVE_DIR, "head_pose_"+word))
        if return_val != 0:
            break

