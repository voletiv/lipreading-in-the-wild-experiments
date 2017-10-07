from process_lrw_params import *
from process_lrw_functions import *

process_lrw(rootDir=LRW_DATA_DIR,startExtracting=True,startSetWordNumber='test/ALWAYS_00057',endSetWordNumber=None,copyTxtFile=False,extractAudioFromMp4=True,dontWriteAudioIfExists=True,extractFramesFromMp4=True,writeFrameImages=True,dontWriteFrameIfExists=True,detectAndSaveMouths=True,dontWriteMouthIfExists=True,verbose=False)
