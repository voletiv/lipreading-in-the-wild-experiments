from process_lrw_functions import *

process_lrw(dataDir=LRW_DATA_DIR,
    saveDir=LRW_SAVE_DIR,
    startExtracting=False,
    startSetWordNumber='test/ATTACKS_00001',
    endSetWordNumber=None,
    copyTxtFile=True,
    extractAudioFromMp4=True,
    dontWriteAudioIfExists=False,
    extractFramesFromMp4=True,
    writeFrameImages=True,
    dontWriteFrameIfExists=False,
    detectAndSaveMouths=True,
    dontWriteMouthIfExists=False,
    verbose=False)

# # DEBUG
# process_lrw(dataDir=LRW_DATA_DIR, saveDir=LRW_SAVE_DIR, startExtracting=True, startSetWordNumber='train/ACCESS_00543', endSetWordNumber=None, copyTxtFile=True, extractAudioFromMp4=True, dontWriteAudioIfExists=False, extractFramesFromMp4=True, writeFrameImages=True, dontWriteFrameIfExists=False, detectAndSaveMouths=True, dontWriteMouthIfExists=False, verbose=True)

# # TEST
# process_lrw(dataDir=LRW_DATA_DIR, saveDir='/home/voletiv/LRW-test', startExtracting=False, startSetWordNumber='val/AFTER_00050', endSetWordNumber='test/AFTERNOON_00001', copyTxtFile=False, extractAudioFromMp4=False, dontWriteAudioIfExists=False, extractFramesFromMp4=False, writeFrameImages=False, dontWriteFrameIfExists=True, detectAndSaveMouths=True, dontWriteMouthIfExists=False, verbose=True)
