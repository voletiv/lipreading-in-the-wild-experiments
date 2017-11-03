import glob
import os
import tqdm

from head_pose_params import *

# Set dataDir
dataDir = LRW_DATA_DIR

################################################
# Find all jpg filenames
################################################


# TO make a new head_pose_jpg_file_names file for each word
def close_and_open_new_f(f, wordDir):
    f.close()
    new_file_name = os.path.join(LRW_SAVE_DIR, "head_pose_jpg_file_names_{0}.txt".format(wordDir.split('/')[-2]))
    f = open(new_file_name, 'w')
    return f, new_file_name

# Starting point, ending point
startSetWordNumber = None
endSetWordNumber = "00020"

if startSetWordNumber is None:
    startExtracting = True
else:
    startExtracting = False

head_pose_jpg_file_names = []

# MAKE .txt FILES OF ALL JPG FILES IN WORD
f = open("head_pose_dummy.txt", 'w')
# word
for wordDir in tqdm.tqdm(sorted(glob.glob(os.path.join(dataDir, '*/')))):
        # print(wordDir, end='\r')
        f, file_name = close_and_open_new_f(f, wordDir)
        head_pose_jpg_file_names.append(file_name)
        # set
        for setDir in sorted(glob.glob(os.path.join(wordDir, '*/'))):
            # print(setDir, end='\r')
            # jpg
            for jpgName in sorted(glob.glob(os.path.join(setDir, '*.jpg'))):
                if startSetWordNumber is not None:
                    if startSetWordNumber in jpgName:
                        startExtracting = True
                if not startExtracting:
                    continue
                if endSetWordNumber is not None:
                    if endSetWordNumber in jpgName:
                        raise KeyboardInterrupt
                if "mouth.jpg" not in jpgName:
                    print(jpgName, end="\r")
                    a = f.write(jpgName + "\n")

f.close()

################################################
# Run gazr_benchmark_head_pose_multiple_frames
################################################

# Call shell command using subprocess
# subprocess.Popen(["/home/voletiv/GitHubRepos/gazr/build/gazr_benchmark_head_pose_multiple_frames", "/home/voletiv/GitHubRepos/lipreading-in-the-wild-experiments/shape-predictor/shape_predictor_68_face_landmarks.dat", "head_pose.txt", ">", "a.txt"])
# "> a.txt" doesn't work

gazr_exe = os.path.join(GAZR_BUILD_DIR, "gazr_benchmark_head_pose_multiple_frames")

for file in head_pose_jpg_file_names:
    head_pose_file_name = os.path.join(LRW_SAVE_DIR, "head_pose_{0}.txt".format(file.split('.')[0].split('_')[-1]))
    command = gazr_exe + " " + SHAPE_DAT_FILE + " " + file + " > " + head_pose_file_name
    os.system(command)

