import sys

from head_pose_functions import *

################################################
# Init
################################################

# Starting point, ending point
startWord = None
startSetWordNumber = None
endWord = None
endSetWordNumber = None
dataDir = LRW_DATA_DIR

for i, v in enumerate(sys.argv):

    if "--startWord" in v or "-sW" in v:
        try:
            startWord = sys.argv[i+1]
        except IndexError:
            print("[ERROR] Please specify where to start from!")
            sys.exit()

    elif "--startSetWordNumber" in v or "-sSWN" in v:
        try:
            startSetWordNumber = sys.argv[i+1]
        except IndexError:
            print("[ERROR] Please specify where to start from!")
            sys.exit()

    elif "--endWord" in v or "-eW" in v:
        try:
            endWord = sys.argv[i+1]
        except IndexError:
            print("[ERROR] Please specify where to end at!")
            sys.exit()

    elif "--endSetWordNumber" in v or "-eSWN" in v:
        try:
            endSetWordNumber = sys.argv[i+1]
        except IndexError:
            print("[ERROR] Please specify where to end at!")
            sys.exit()

    elif "--dataDir" in v or '-d' in v:
        try:
            dataDir = sys.argv[i+1]
        except IndexError:
            print("[ERROR] Please specify dataDir properly!")
            sys.exit()



################################################
# Find all jpg filenames
################################################

write_frame_jpg_file_names_in_txt_file(dataDir, startWord=startWord, startSetWordNumber=startSetWordNumber, endWord=endWord, endSetWordNumber=endSetWordNumber)
