from process_lrw_params import *
from process_lrw_functions import *

process_lrw(dataDir=LRW_DATA_DIR,
    saveDir=LRW_SAVE_DIR,
    startExtracting=True,
    startSetWordNumber='test/ALWAYS_00057',
    endSetWordNumber=None,
    copyTxtFile=True,
    extractAudioFromMp4=True,
    dontWriteAudioIfExists=True,
    extractFramesFromMp4=True,
    writeFrameImages=True,
    dontWriteFrameIfExists=True,
    detectAndSaveMouths=True,
    dontWriteMouthIfExists=True,
    verbose=False)

# # DEBUG
# process_lrw(dataDir=LRW_DATA_DIR, saveDir=LRW_SAVE_DIR, startExtracting=True, startSetWordNumber='train/ACCESS_00543', endSetWordNumber=None, copyTxtFile=True, extractAudioFromMp4=True, dontWriteAudioIfExists=False, extractFramesFromMp4=True, writeFrameImages=True, dontWriteFrameIfExists=False, detectAndSaveMouths=True, dontWriteMouthIfExists=False, verbose=True)

# # TEST
# process_lrw(dataDir=LRW_DATA_DIR, saveDir='/home/voletiv/LRW-test', startExtracting=False, startSetWordNumber='train/ACROSS_00460', endSetWordNumber='train/ACROSS_00461', copyTxtFile=False, extractAudioFromMp4=False, dontWriteAudioIfExists=False, extractFramesFromMp4=False, writeFrameImages=False, dontWriteFrameIfExists=True, detectAndSaveMouths=True, dontWriteMouthIfExists=False, verbose=True)
