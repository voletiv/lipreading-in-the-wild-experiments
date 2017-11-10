import sys

from head_pose_functions import *

################################################
# Init
################################################

# Starting point, ending point
dataDir = LRW_DATA_DIR
startWord = None
startSetWordNumber = None
endWord = None
endSetWordNumber = None
append_to_file = True

for i, v in enumerate(sys.argv):

    if "--startWord" in v or "-sW" in v:
        try:
            startWord = sys.argv[i+1]
        except IndexError:
            print("[ERROR] Please specify where to start from!")
            sys.exit()

    if "--startSetWordNumber" in v or "-sSWN" in v:
        try:
            startSetWordNumber = sys.argv[i+1]
        except IndexError:
            print("[ERROR] Please specify where to start from!")
            sys.exit()

    if "--endWord" in v or "-eW" in v:
        try:
            endWord = sys.argv[i+1]
        except IndexError:
            print("[ERROR] Please specify where to end at!")
            sys.exit()

    if "--endSetWordNumber" in v or "-eSWN" in v:
        try:
            endSetWordNumber = sys.argv[i+1]
        except IndexError:
            print("[ERROR] Please specify where to end at!")
            sys.exit()

    if "--dataDir" in v or '-d' in v:
        try:
            dataDir = sys.argv[i+1]
        except IndexError:
            print("[ERROR] Please specify dataDir properly!")
            sys.exit()

    if "-w" in v:
        append_to_file = False


################################################
# Find all jpg filenames
################################################

write_frame_jpg_file_names_in_txt_file(dataDir, startWord=startWord, startSetWordNumber=startSetWordNumber, endWord=endWord, endSetWordNumber=endSetWordNumber, append_to_file=append_to_file)
