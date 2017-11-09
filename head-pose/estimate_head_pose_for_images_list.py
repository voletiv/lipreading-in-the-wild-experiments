import sys

from head_pose_functions import *

sys.path.append(DEEPGAZE_EXAMPLES_DIR)

from ex_cnn_head_pose_estimation_images_list import *

for file in tqdm.tqdm(sorted(glob.glob(os.path.join(LRW_DATA_DIR, 'head_pose_jpg_file_names*')))):
    word = file.split('/')[-1].split('.')[-2].split('_')[-1]
    cnn_head_pose_estimation_images_list(file, save_npy=True, npy_file_name="head_pose_"+word)


