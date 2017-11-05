from head_pose_functions import *

# Set dataDir
dataDir = LRW_DATA_DIR

################################################
# Find all jpg filenames
################################################

# Starting point, ending point
startSetWordNumber = None
endSetWordNumber = None

write_frame_jpg_file_names_in_txt_file(dataDir, startSetWordNumber=startSetWordNumber, endSetWordNumber=endSetWordNumber)

################################################
# Run gazr_benchmark_head_pose_multiple_frames
################################################

run_dlib_head_pose_estimator(dataDir)
