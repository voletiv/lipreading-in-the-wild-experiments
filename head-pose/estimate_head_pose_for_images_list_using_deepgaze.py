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



#########################################################
# DEBUG - NUM OF SAMPLES IN VAL
#########################################################

image_file_names = []
for file in sorted(glob.glob(os.path.join(LRW_SAVE_DIR, 'head_pose_jpg_file_names*'))):
    with open(file) as f:
        for line in f:
            if 'val' in line:
                    image_file_names.append(line.rstrip())

from deepgaze.head_pose_estimation import CnnHeadPoseEstimator

sess = tf.Session() #Launch the graph in a session.
my_head_pose_estimator = CnnHeadPoseEstimator(sess) #Head pose estimation object

# Load the weights from the configuration folders
DEEPGAZE_EXAMPLES_DIR = '/shared/fusor/home/voleti.vikram/deepgaze/examples'
my_head_pose_estimator.load_roll_variables(os.path.realpath(os.path.join(DEEPGAZE_EXAMPLES_DIR, "../etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf")))
my_head_pose_estimator.load_pitch_variables(os.path.realpath(os.path.join(DEEPGAZE_EXAMPLES_DIR, "../etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf")))
my_head_pose_estimator.load_yaw_variables(os.path.realpath(os.path.join(DEEPGAZE_EXAMPLES_DIR, "../etc/tensorflow/head_pose/yaw/cnn_cccdd_30k")))

poses = np.zeros((1, 3))

try:
    prev_word = "dummy"
    for image_file in tqdm.tqdm(image_file_names):
        word = image_file.split('/')[-1].split('.')[0].split('_')[0]
        if word != prev_word:
            np.save(os.path.join(LRW_SAVE_DIR, "head_pose_"+prev_word), poses)
            prev_word = word
            poses = np.empty((0, 3))
        #Read the image with OpenCV
        image = cv2.imread(image_file)
        # Get the angles for roll, pitch and yaw
        roll = my_head_pose_estimator.return_roll(image)  # Evaluate the roll angle using a CNN
        pitch = my_head_pose_estimator.return_pitch(image)  # Evaluate the pitch angle using a CNN
        yaw = my_head_pose_estimator.return_yaw(image)  # Evaluate the yaw angle using a CNN
        poses = np.vstack((poses, [(roll[0,0,0])/25, pitch[0,0,0]/45, yaw[0,0,0]/100]))
except KeyboardInterrupt:
    print("\n\nCtrl+C was pressed!\n\n")
    return_val = 1

