import sys
sys.setrecursionlimit(10000)

import os
os.environ['KERAS_BACKEND']='tensorflow'

import cv2
import glob
import numpy as np
import time
import tqdm

############################################################################################################

#############################           CNN         ####################################

############################################################################################################

from keras.layers import Input, ZeroPadding3D, Conv3D, BatchNormalization, Activation, MaxPooling3D, Permute, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense, MaxPooling1D, Conv1D, AveragePooling1D, Flatten
from keras.models import Model

from keras import backend as K
K.set_image_dim_ordering('th')

############################################################################################################

#############################           CNN RESNET 50       ####################################

############################################################################################################

input1 = Input(shape=(1, 29, 112, 112))
pad1= ZeroPadding3D((1, 3, 3))(input1)
conv1 = Conv3D(64, (5, 7, 7), name="conv1", strides=(1, 2, 2), padding="valid")(pad1)
B1 = BatchNormalization(axis=1)(conv1)
act1 = Activation('relu')(B1)
padm1= ZeroPadding3D((0, 1, 1))(act1)
m1 = MaxPooling3D((1,3,3), strides=(1,2,2))(padm1)
perm1= Permute(dims=(2,1,3,4))(m1)
Flat1= Reshape((27,64*28*28))(perm1)

lin1=TimeDistributed(Dense(384))(Flat1)
B_lin1 = BatchNormalization(axis=-1)(lin1)
act_lin1 = Activation('relu')(B_lin1)

lin2=TimeDistributed(Dense(384))(act_lin1)
B_lin2 = BatchNormalization(axis=-1)(lin2)
act_lin2 = Activation('relu')(B_lin2)

lin3=TimeDistributed(Dense(256))(act_lin2)
B_lin3 = BatchNormalization(axis=-1)(lin3)
act_lin3 = Activation('relu')(B_lin3)

conv2 = Conv1D(512, 5, name="conv2", strides=2, padding="valid")(act_lin3)
B_conv2 = BatchNormalization(axis=-1)(conv2)
act_conv2 = Activation('relu')(B_conv2)
m2 = MaxPooling1D(2, strides=2)(act_conv2)

conv3 = Conv1D(1024, 3, name="conv3", strides=2, padding="valid")(m2)
B_conv3 = BatchNormalization(axis=-1)(conv3)
act_conv3 = Activation('relu')(B_conv3)
#m3 = MaxPooling1D(3, strides=2)(act_conv3)

av1=AveragePooling1D(2, strides=2)(act_conv3)

av1R = Reshape((1024,))(av1)
denseOutput = Model(inputs=[input1], outputs=[av1R])

Flat2= Flatten()(av1)
lin3=Dense(500)(Flat2)
softy = Activation('softmax')(lin3)
model1 = Model(inputs=[input1], outputs=[softy])
model1.summary()

model1.load_weights('/shared/magnetar/datasets/LipReading/stafylakis_lipreader_feature/lipreader/saved_model_weights_stafylakis_important_[working]/save_1_gpu_model.hdf5')

############################################################################################################

#############################           Main Pipeline       ####################################

############################################################################################################

############################################################################################################

#############################           Validation      ####################################

############################################################################################################

x_max_len=29
batch_size=100;
flag1=1
numi4=0
nb_classes=500
# charlist=open('charsList.txt').readlines()
def abhishek_generator(allFilenames,batch_size,skip=0,shuffle1=0):
    if shuffle1==1:
        allFilenames=shuffle(allFilenames)
    while 1:
        for i0 in range(0,len(allFilenames),batch_size):
            if i0 < skip*batch_size:
                continue
            numi4=0
            X_val=np.zeros(shape=(batch_size,x_max_len,112,112))
            Y_val=np.zeros(shape=(batch_size,nb_classes))
            for i1 in range(0,batch_size):
                ii1=i0+i1
                #print ii1
                try:
                    ii=allFilenames[ii1].split('/')[-2]
                    dataii=allFilenames[ii1]
                except:
                    continue
                flagi1=[]
                for path, subdirs, filesi1 in os.walk(dataii):
                    flagi1.append(filesi1)
                if len(flagi1[0])<1:
                    continue
                rois=np.zeros(shape=(x_max_len,112,112))
                splitD=dataii.split('/')
                rand1=random.randint(0, 4)
                rand2=random.randint(0, 4)
                rand3=random.randint(0,1)
                for indxN in range(0,len(filesi1)):
                    #img_path =(os.path.join(path, name))
                    img_path = dataii+splitD[-2]+'_'+str(indxN)+'.jpg'
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = img[4-rand1:64-rand1,4-rand2:64-rand2]
                    img1 = cv2.resize(img, (112, 112)).astype(np.float32)
                    if rand3==1:
                        img1=cv2.flip(img1,1)
                    img1 = (1.0*img1)/255
                    img1 -= 0.5
                    rois[indxN]=img1
                X_val[numi4]=rois
                Y_val[numi4][all_words.index(splitD[-4])]=1
                numi4+=1
            X_val=X_val[:numi4]
            X_val=np.expand_dims(X_val,axis=1)
            Y_val=Y_val[:numi4]
            yield (X_val,Y_val)


def load_lrw_vocab_list(LRW_VOCAB_LIST_FILE):
    lrw_vocab = []
    with open(LRW_VOCAB_LIST_FILE) as f:
        for line in f:
            word = line.rstrip().split()[-1]
            lrw_vocab.append(word)
    return lrw_vocab

LRW_DIR = '/shared/fusor/home/voleti.vikram/lipreading-in-the-wild-experiments/'

LRW_VOCAB_LIST_FILE = os.path.join(LRW_DIR, 'lrw_vocabulary.txt')

LRW_VOCAB = load_lrw_vocab_list(LRW_VOCAB_LIST_FILE)

LRW_VOCAB_SIZE = len(LRW_VOCAB)


def vikram_generator(lrw_word_set_num_txt_file_names, batch_size, skip=0, shuffle=False, random_crop=True, random_flip=True):
    allTxtFilenames = np.array(lrw_word_set_num_txt_file_names)
    x_max_len=29
    nb_classes=500
    if shuffle:
        allTxtFilenames=np.random.shuffle(allTxtFilenames)
    while 1:
        n_batches = len(allTxtFilenames) // batch_size + 1
        # For each batch
        for batch in range(n_batches):
            if batch < skip:
                continue
            this_batch_size = batch_size
            if (batch+1)*batch_size > len(allTxtFilenames):
                this_batch_size = len(allTxtFilenames) - batch*batch_size
            X_val = np.zeros(shape=(this_batch_size, x_max_len, 112, 112))
            Y_val = np.zeros(shape=(this_batch_size, nb_classes))
            for i in range(this_batch_size):
                rois = np.zeros(shape=(x_max_len, 112, 112))
                if random_crop:
                    crop_rand1 = random.randint(0, 8)
                    crop_rand2 = random.randint(0, 8)
                else:
                    crop_rand1 = 4
                    crop_rand2 = 4
                if random_flip:
                    flip_rand = random.randint(0,1)
                else:
                    flip_rand = False
                word_txt_file = allTxtFilenames[batch*batch_size+i]
                for indxN, img_path in enumerate(sorted(glob.glob('.'.join(word_txt_file.split('.')[:-1]) + '*mouth*.jpg'))):
                    if indxN+1 > x_max_len:
                        continue
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img1 = img[crop_rand1:crop_rand1+112, crop_rand2:crop_rand2+112]
                    if flip_rand:
                        img1 = cv2.flip(img1, 1)
                    img1 = img1 / 255.
                    img1 -= 0.5
                    rois[indxN] = img1
                X_val[i] = rois
                Y_val[i][LRW_VOCAB.index(word_txt_file.split('/')[-3])] = 1
            X_val = np.expand_dims(X_val, axis=1)
            yield (X_val,Y_val)


############################################################################################################

#############################           Read        ####################################

############################################################################################################


# root='/shared/magnetar/home/abhishek/new_LRW_may17/'
root='/shared/fusor/home/voleti.vikram/LRW-abhishek/'

# all_words = next(os.walk(root))[1]
# all_words = sorted(all_words)

# allFilenames_train=[]
# for i0 in tqdm.tqdm(range(0,len(all_words))):
#     j0=all_words[i0]
#     # print("i0=",i0," percent_complete= ",float(i0*100)/len(all_words),"%")
#     path_mp4_files=root+j0+'/train/'
#     mp4_files = next(os.walk(path_mp4_files))[1]
#     mp4_files = sorted(mp4_files)
#     for i1 in range(0,len(mp4_files)):
#         j1=mp4_files[i1]
#         root_image_path=root+j0+'/train/'+j1+'/'
#         allFilenames_train.append(root_image_path)

# allFilenames_val=[]
# for i0 in tqdm.tqdm(range(0,len(all_words))):
#     j0=all_words[i0]
#     # print("i0=",i0," percent_complete= ",float(i0*100)/len(all_words),"%")
#     path_mp4_files=root+j0+'/val/'
#     mp4_files = next(os.walk(path_mp4_files))[1]
#     mp4_files = sorted(mp4_files)
#     for i1 in tqdm.tqdm(range(0,len(mp4_files))):
#         j1=mp4_files[i1]
#         root_image_path=root+j0+'/val/'+j1+'/'
#         allFilenames_val.append(root_image_path)

# allFilenames_test=[]
# for i0 in tqdm.tqdm(range(0,len(all_words))):
#     j0=all_words[i0]
#     # print("i0=",i0," percent_complete= ",float(i0*100)/len(all_words),"%")
#     path_mp4_files=root+j0+'/test/'
#     mp4_files = next(os.walk(path_mp4_files))[1]
#     mp4_files = sorted(mp4_files)
#     for i1 in tqdm.tqdm(range(0,len(mp4_files))):
#         j1=mp4_files[i1]
#         root_image_path=root+j0+'/test/'+j1+'/'
#         allFilenames_test.append(root_image_path)

# MY WAY
import sys
sys.path.append('/shared/fusor/home/voleti.vikram/lipreading-in-the-wild-experiments/assessor/')
from assessor_functions import *
lrw_word_set_num_txt_file_names_train = read_lrw_word_set_num_file_names(collect_type='train', collect_by='sample')
lrw_word_set_num_txt_file_names_val = read_lrw_word_set_num_file_names(collect_type='val', collect_by='sample')
lrw_word_set_num_txt_file_names_test = read_lrw_word_set_num_file_names(collect_type='test', collect_by='sample')


# ############################################################################################################

# #############################           Train           ####################################

# ############################################################################################################
# batchSz=64
# trainSteps=int(len((allFilenames_train)))/batchSz
# valSteps=int(len((allFilenames_val)))/batchSz
# historyLSTM=model1.fit_generator(my_generator(allFilenames_train,batch_size=batchSz,shuffle1=1), steps_per_epoch=trainSteps, validation_data=my_generator(allFilenames_val,batch_size=batchSz,shuffle1=0),nb_epoch=300, verbose=1, nb_worker=1,validation_steps=valSteps, callbacks=[checkpoint])


############################################################################################################

#############################                   Prediction              ####################################

############################################################################################################

# TRAIN
batch_size = 100
trainY = np.empty((0, 500))
trainDense = np.empty((0, 1024))
trainSoftmax = np.empty((0, 500))
npz_name_train = "/shared/fusor/home/voleti.vikram/LRW_train_dense_softmax_y"
to_skip = 0
if os.path.exists(npz_name_train+".npz"):
    npz_train = np.load(npz_name_train+".npz")
    trainDense = npz_train["trainDense"]
    trainSoftmax = npz_train["trainSoftmax"]
    trainY = npz_train["trainY"]
    to_skip = len(trainDense)//batch_size

k = vikram_generator(lrw_word_set_num_txt_file_names_train, batch_size=batch_size, skip=to_skip, shuffle=False, random_crop=False, random_flip=False)
n_steps = len(lrw_word_set_num_txt_file_names_train)//batch_size
for i in tqdm.tqdm(range(to_skip, n_steps), initial=to_skip, total=n_steps):
    (X, Y) = next(k)
    trainY = np.vstack((trainY, Y))
    softmax = model1.predict(X)
    trainSoftmax = np.vstack((trainSoftmax, softmax))
    dense = denseOutput.predict(X)
    trainDense = np.vstack((trainDense, dense))
    np.savez(npz_name_train, trainDense=trainDense, trainSoftmax=trainSoftmax, trainY=trainY)


# VAL
batch_size = 100
valY = np.empty((0, 500))
valDense = np.empty((0, 1024))
valSoftmax = np.empty((0, 500))
npz_name_val = "/shared/fusor/home/voleti.vikram/LRW_val_dense_softmax_y"
to_skip = 0
if os.path.exists(npz_name_val+".npz"):
    npz_val = np.load(npz_name_val+".npz")
    valDense = npz_val["valDense"]
    valSoftmax = npz_val["valSoftmax"]
    valY = npz_val["valY"]
    to_skip = len(npz_val["valDense"])//batch_size

k = vikram_generator(lrw_word_set_num_txt_file_names_val, batch_size=batch_size, skip=to_skip, shuffle=False, random_crop=False, random_flip=False)
n_steps = len(lrw_word_set_num_txt_file_names_val)//batch_size
for i in tqdm.tqdm(range(to_skip, n_steps), initial=to_skip, total=n_steps):
    (X, Y) = next(k)
    valY = np.vstack((valY, Y))
    softmax = model1.predict(X)
    valSoftmax = np.vstack((valSoftmax, softmax))
    dense = denseOutput.predict(X)
    valDense = np.vstack((valDense, dense))
    np.savez("/shared/fusor/home/voleti.vikram/LRW_val_dense_softmax_y", valDense=valDense, valSoftmax=valSoftmax, valY=valY)


# TEST
testDense = np.empty((0, 1024))
testSoftmax = np.empty((0, 500))
testY = np.empty((0, 500))
npz_name_test = "/shared/fusor/home/voleti.vikram/LRW_test_dense_softmax_y"
to_skip = 0
if os.path.exists(npz_name_test+".npz"):
    npz_test = np.load(npz_name_test+".npz")
    testDense = npz_test["testDense"]
    testSoftmax = npz_test["testSoftmax"]
    testY = npz_test["testY"]
    to_skip = len(npz_test["valDense"])//batch_size

k = vikram_generator(lrw_word_set_num_txt_file_names_test, batch_size=batch_size, skip=to_skip, shuffle=False, random_crop=False, random_flip=False)
n_steps = len(lrw_word_set_num_txt_file_names_test)//batch_size
for i in tqdm.tqdm(range(to_skip, n_steps), initial=to_skip, total=n_steps):
    (X, Y) = next(k)
    testY = np.vstack((testY, Y))
    softmax = model1.predict(X)
    testSoftmax = np.vstack((testSoftmax, softmax))
    dense = denseOutput.predict(X)
    testDense = np.vstack((testDense, dense))
    np.savez("/shared/fusor/home/voleti.vikram/LRW_test_dense_softmax_y", testDense=testDense, testSoftmax=testSoftmax, testY=testY)

print("Done")
