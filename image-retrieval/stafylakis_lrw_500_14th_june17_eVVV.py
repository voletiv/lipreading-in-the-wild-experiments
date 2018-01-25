import sys
sys.setrecursionlimit(10000)
sys.path.append('/shared/fusor/home/voleti.vikram/lipreading-in-the-wild-experiments/assessor/')

import os
os.environ['KERAS_BACKEND']='tensorflow'

import cv2
import glob
import numpy as np
import time
import tqdm

from assessor_functions import *
from checkpoint_and_make_plots import *

############################################################################################################

#############################           CNN         ####################################

############################################################################################################

from keras.layers import Input, ZeroPadding3D, Conv3D, BatchNormalization, Activation, MaxPooling3D, Permute, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense, MaxPooling1D, Conv1D, AveragePooling1D, Flatten
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Model
from keras.utils import multi_gpu_model

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

# model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

parallel_model = multi_gpu_model(model1, gpus=4)

parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model1.load_weights('/shared/magnetar/datasets/LipReading/stafylakis_lipreader_feature/lipreader/saved_model_weights_stafylakis_important_[working]/save_1_gpu_model.hdf5')

############################################################################################################

#############################           Main Pipeline       ####################################

############################################################################################################

############################################################################################################

#############################           Validation      ####################################

############################################################################################################

x_max_len = 29
batch_size = 100;
flag1 = 1
numi4 = 0
nb_classes = 500

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
    x_max_len = 29
    nb_classes = 500
    n_batches = len(allTxtFilenames) // batch_size
    while 1:
        if shuffle:
            np.random.shuffle(allTxtFilenames)
        # For each batch
        for batch in range(n_batches):
            if batch < skip:
                continue
            this_batch_size = batch_size
            if (batch+1)*batch_size > len(allTxtFilenames):
                this_batch_size = len(allTxtFilenames) - batch*batch_size
            X = np.zeros(shape=(this_batch_size, x_max_len, 112, 112))
            Y = np.zeros(shape=(this_batch_size, nb_classes))
            for i in range(this_batch_size):
                rois = np.zeros(shape=(x_max_len, 112, 112))
                if random_crop:
                    crop_rand1 = np.random.randint(0, 8)
                    crop_rand2 = np.random.randint(0, 8)
                else:
                    crop_rand1 = 4
                    crop_rand2 = 4
                if random_flip:
                    flip_rand = np.random.randint(0,1)
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
                X[i] = rois
                Y[i][LRW_VOCAB.index(word_txt_file.split('/')[-3])] = 1
            X = np.expand_dims(X, axis=1)
            yield (X,Y)


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
lrw_word_set_num_txt_file_names_train = read_lrw_word_set_num_file_names(collect_type='train', collect_by='sample')
lrw_word_set_num_txt_file_names_val = read_lrw_word_set_num_file_names(collect_type='val', collect_by='sample')
lrw_word_set_num_txt_file_names_test = read_lrw_word_set_num_file_names(collect_type='test', collect_by='sample')

############################################################################################################

#############################           Make Spl Order of Files         ####################################

############################################################################################################

lrw_word_set_num_txt_file_names_by_word_train = read_lrw_word_set_num_file_names(collect_type='train', collect_by='vocab_word')
lrw_word_set_num_txt_file_names_by_word_val = read_lrw_word_set_num_file_names(collect_type='val', collect_by='vocab_word')

# Choose sequential samples
spl_filenames_train = []
# Train From LRW_TRAIN: make 400 samples per word, with offset 400
offset = 400
samples_per_word = 400
for w in range(500):
    for i in range(samples_per_word):
        spl_filenames_train.append(lrw_word_set_num_txt_file_names_by_word_train[w][offset + i])

# Train From LRW_VAL: make 25 samples per word, with offset 25
offset = 25
samples_per_word = 25
for w in range(500):
    for i in range(samples_per_word):
        spl_filenames_train.append(lrw_word_set_num_txt_file_names_by_word_val[w][offset + i])

spl_filenames_val = []
# Val form LRW_TRAIN:
offset = 0
samples_per_word = 25
for w in range(500):
    for i in range(samples_per_word):
        spl_filenames_val.append(lrw_word_set_num_txt_file_names_by_word_train[w][offset + i])

# Val form LRW_VAL:
offset = 0
samples_per_word = 25
for w in range(500):
    for i in range(samples_per_word):
        spl_filenames_val.append(lrw_word_set_num_txt_file_names_by_word_val[w][offset + i])


############################################################################################################

#############################           Train           ####################################################

############################################################################################################

batch_size = 128
epochs = 1000

train_generator = vikram_generator(spl_filenames_train, batch_size=batch_size, shuffle=True, random_crop=True, random_flip=True)
val_generator = vikram_generator(spl_filenames_val, batch_size=batch_size, shuffle=False, random_crop=False, random_flip=False)

train_steps = len((spl_filenames_train)) // batch_size
val_steps = len((spl_filenames_val)) // batch_size

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), patience=5, verbose=1)

early_stopper = EarlyStopping(min_delta=0.001, patience=50)

model_name = 'LRW_LIPREADER_TRAIN400offset400_VAL25OFFSET25'
save_dir = os.path.join("/shared/fusor/home/voleti.vikram", model_name)
checkpointAndMakePlots = CheckpointAndMakePlots(file_name_pre=model_name, save_dir=save_dir)

history = model1.fit_generator(train_generator, steps_per_epoch=train_steps, epochs=epochs, verbose=1,
                                       callbacks=[lr_reducer, early_stopper, checkpointAndMakePlots],
                                       validation_data=val_generator, validation_steps=val_steps)




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


###################################################
# COMBINE
###################################################

samples_per_word = 200
train_D_200samplesPerWord = np.zeros((samples_per_word*500, 1024))
train_S_200samplesPerWord = np.zeros((samples_per_word*500, 500))
train_Y_200samplesPerWord = np.zeros((samples_per_word*500), dtype=int)

correct_lrw_softmax_argmax_file = '/shared/fusor/home/voleti.vikram/lipreading-in-the-wild-experiments/assessor/correct_lrw_softmax_argmax.txt'
correct_lrw_softmax_argmax = []
with open(correct_lrw_softmax_argmax_file) as f:
    for line in f:
        correct_lrw_softmax_argmax.append(int(line.rstrip()))

# 0 to 230
train_DSY_200samplesPerWord_0to232words = np.load('LRW_train_dense_softmax_y_200samplesPerWord_0offset_0to232words.npz')
train_D_200samplesPerWord_0to230words = train_DSY_200samplesPerWord_0to232words['trainDense'][:230*200]
train_S_200samplesPerWord_0to230words = train_DSY_200samplesPerWord_0to232words['trainSoftmax'][:230*200]


## 200 samples per word
for w in range(230):
    train_D_200samplesPerWord[w*samples_per_word:(w+1)*samples_per_word] = train_D_200samplesPerWord_0to230words[w*200:(w*200+samples_per_word)]
    train_S_200samplesPerWord[w*samples_per_word:(w+1)*samples_per_word] = train_S_200samplesPerWord_0to230words[w*200:(w*200+samples_per_word)]
    train_Y_200samplesPerWord[w*samples_per_word:(w+1)*samples_per_word] = correct_lrw_softmax_argmax[w]

# 230 to 500
train_DSY_20samplesPerWord_0offset_230to500words = np.load('LRW_train_dense_softmax_y_20samplesPerWord_0offset_230to500words.npz')
train_D_20samplesPerWord_0offset_230to500words = train_DSY_20samplesPerWord_0offset_230to500words['trainDense']
train_S_20samplesPerWord_0offset_230to500words = train_DSY_20samplesPerWord_0offset_230to500words['trainSoftmax']

train_DSY_20samplesPerWord_20offset_230to500words = np.load('LRW_train_dense_softmax_y_20samplesPerWord_20offset_230to500words.npz')
train_D_20samplesPerWord_20offset_230to500words = train_DSY_20samplesPerWord_20offset_230to500words['trainDense']
train_S_20samplesPerWord_20offset_230to500words = train_DSY_20samplesPerWord_20offset_230to500words['trainSoftmax']

train_DSY_20samplesPerWord_40offset_230to500words = np.load('LRW_train_dense_softmax_y_20samplesPerWord_40offset_230to500words.npz')
train_D_20samplesPerWord_40offset_230to500words = train_DSY_20samplesPerWord_40offset_230to500words['trainDense']
train_S_20samplesPerWord_40offset_230to500words = train_DSY_20samplesPerWord_40offset_230to500words['trainSoftmax']

train_DSY_20samplesPerWord_60offset_230to500words = np.load('LRW_train_dense_softmax_y_20samplesPerWord_60offset_230to500words.npz')
train_D_20samplesPerWord_60offset_230to500words = train_DSY_20samplesPerWord_60offset_230to500words['trainDense']
train_S_20samplesPerWord_60offset_230to500words = train_DSY_20samplesPerWord_60offset_230to500words['trainSoftmax']

train_DSY_20samplesPerWord_80offset_230to500words = np.load('LRW_train_dense_softmax_y_20samplesPerWord_80offset_230to500words.npz')
train_D_20samplesPerWord_80offset_230to500words = train_DSY_20samplesPerWord_80offset_230to500words['trainDense']
train_S_20samplesPerWord_80offset_230to500words = train_DSY_20samplesPerWord_80offset_230to500words['trainSoftmax']

train_DSY_20samplesPerWord_100offset_230to500words = np.load('LRW_train_dense_softmax_y_20samplesPerWord_100offset_230to500words.npz')
train_D_20samplesPerWord_100offset_230to500words = train_DSY_20samplesPerWord_100offset_230to500words['trainDense']
train_S_20samplesPerWord_100offset_230to500words = train_DSY_20samplesPerWord_100offset_230to500words['trainSoftmax']

train_DSY_20samplesPerWord_120offset_230to500words = np.load('LRW_train_dense_softmax_y_20samplesPerWord_120offset_230to500words.npz')
train_D_20samplesPerWord_120offset_230to500words = train_DSY_20samplesPerWord_120offset_230to500words['trainDense']
train_S_20samplesPerWord_120offset_230to500words = train_DSY_20samplesPerWord_120offset_230to500words['trainSoftmax']

train_DSY_20samplesPerWord_140offset_230to500words = np.load('LRW_train_dense_softmax_y_20samplesPerWord_140offset_230to500words.npz')
train_D_20samplesPerWord_140offset_230to500words = train_DSY_20samplesPerWord_140offset_230to500words['trainDense']
train_S_20samplesPerWord_140offset_230to500words = train_DSY_20samplesPerWord_140offset_230to500words['trainSoftmax']

train_DSY_20samplesPerWord_160offset_230to500words = np.load('LRW_train_dense_softmax_y_20samplesPerWord_160offset_230to500words.npz')
train_D_20samplesPerWord_160offset_230to500words = train_DSY_20samplesPerWord_160offset_230to500words['trainDense']
train_S_20samplesPerWord_160offset_230to500words = train_DSY_20samplesPerWord_160offset_230to500words['trainSoftmax']

train_DSY_20samplesPerWord_180offset_230to500words = np.load('LRW_train_dense_softmax_y_20samplesPerWord_180offset_230to500words.npz')
train_D_20samplesPerWord_180offset_230to500words = train_DSY_20samplesPerWord_180offset_230to500words['trainDense']
train_S_20samplesPerWord_180offset_230to500words = train_DSY_20samplesPerWord_180offset_230to500words['trainSoftmax']

for w in range(230, 500):
    train_D_200samplesPerWord[(w*samples_per_word+00):(w*samples_per_word+20)] = train_D_20samplesPerWord_0offset_230to500words[(w-230)*20:((w-230+1)*20)]
    train_D_200samplesPerWord[(w*samples_per_word+20):(w*samples_per_word+40)] = train_D_20samplesPerWord_20offset_230to500words[(w-230)*20:((w-230+1)*20)]
    train_D_200samplesPerWord[(w*samples_per_word+40):(w*samples_per_word+60)] = train_D_20samplesPerWord_60offset_230to500words[(w-230)*20:((w-230+1)*20)]
    train_D_200samplesPerWord[(w*samples_per_word+60):(w*samples_per_word+80)] = train_D_20samplesPerWord_60offset_230to500words[(w-230)*20:((w-230+1)*20)]
    train_D_200samplesPerWord[(w*samples_per_word+80):(w*samples_per_word+100)] = train_D_20samplesPerWord_80offset_230to500words[(w-230)*20:((w-230+1)*20)]
    train_D_200samplesPerWord[(w*samples_per_word+100):(w*samples_per_word+120)] = train_D_20samplesPerWord_100offset_230to500words[(w-230)*20:((w-230+1)*20)]
    train_D_200samplesPerWord[(w*samples_per_word+120):(w*samples_per_word+140)] = train_D_20samplesPerWord_120offset_230to500words[(w-230)*20:((w-230+1)*20)]
    train_D_200samplesPerWord[(w*samples_per_word+140):(w*samples_per_word+160)] = train_D_20samplesPerWord_140offset_230to500words[(w-230)*20:((w-230+1)*20)]
    train_D_200samplesPerWord[(w*samples_per_word+160):(w*samples_per_word+180)] = train_D_20samplesPerWord_160offset_230to500words[(w-230)*20:((w-230+1)*20)]
    train_D_200samplesPerWord[(w*samples_per_word+180):(w*samples_per_word+200)] = train_D_20samplesPerWord_180offset_230to500words[(w-230)*20:((w-230+1)*20)]
    train_S_200samplesPerWord[(w*samples_per_word+00):(w*samples_per_word+20)] = train_S_20samplesPerWord_0offset_230to500words[(w-230)*20:((w-230+1)*20)]
    train_S_200samplesPerWord[(w*samples_per_word+20):(w*samples_per_word+40)] = train_S_20samplesPerWord_20offset_230to500words[(w-230)*20:((w-230+1)*20)]
    train_S_200samplesPerWord[(w*samples_per_word+40):(w*samples_per_word+60)] = train_S_20samplesPerWord_40offset_230to500words[(w-230)*20:((w-230+1)*20)]
    train_S_200samplesPerWord[(w*samples_per_word+60):(w*samples_per_word+80)] = train_S_20samplesPerWord_60offset_230to500words[(w-230)*20:((w-230+1)*20)]
    train_S_200samplesPerWord[(w*samples_per_word+80):(w*samples_per_word+100)] = train_S_20samplesPerWord_80offset_230to500words[(w-230)*20:((w-230+1)*20)]
    train_S_200samplesPerWord[(w*samples_per_word+100):(w*samples_per_word+120)] = train_S_20samplesPerWord_100offset_230to500words[(w-230)*20:((w-230+1)*20)]
    train_S_200samplesPerWord[(w*samples_per_word+120):(w*samples_per_word+140)] = train_S_20samplesPerWord_120offset_230to500words[(w-230)*20:((w-230+1)*20)]
    train_S_200samplesPerWord[(w*samples_per_word+140):(w*samples_per_word+160)] = train_S_20samplesPerWord_140offset_230to500words[(w-230)*20:((w-230+1)*20)]
    train_S_200samplesPerWord[(w*samples_per_word+160):(w*samples_per_word+180)] = train_S_20samplesPerWord_160offset_230to500words[(w-230)*20:((w-230+1)*20)]
    train_S_200samplesPerWord[(w*samples_per_word+180):(w*samples_per_word+200)] = train_S_20samplesPerWord_180offset_230to500words[(w-230)*20:((w-230+1)*20)]
    train_Y_200samplesPerWord[w*samples_per_word:(w+1)*samples_per_word] = correct_lrw_softmax_argmax[w]

np.savez('LRW_train_dense_softmax_y_200samplesPerWord',
         lrw_train_dense=train_D_200samplesPerWord,
         lrw_train_softmax=train_S_200samplesPerWord,
         lrw_correct_one_hot_arg=train_Y_200samplesPerWord)







## 50 samples per word

for w in range(230):
    train_D_50samplesPerWord[w*50:(w+1)*50] = train_D_200samplesPerWord_0to230words[w*200:(w*200+50)]
    train_S_50samplesPerWord[w*50:(w+1)*50] = train_S_200samplesPerWord_0to230words[w*200:(w*200+50)]
    train_Y_50samplesPerWord[w*50:(w+1)*50] = correct_lrw_softmax_argmax[w]

# 230 to 500
train_DSY_20samplesPerWord_0offset_230to500words = np.load('LRW_train_dense_softmax_y_20samplesPerWord_0offset_230to500words.npz')
train_D_20samplesPerWord_0offset_230to500words = train_DSY_20samplesPerWord_0offset_230to500words['trainDense']
train_S_20samplesPerWord_0offset_230to500words = train_DSY_20samplesPerWord_0offset_230to500words['trainSoftmax']

train_DSY_20samplesPerWord_20offset_230to500words = np.load('LRW_train_dense_softmax_y_20samplesPerWord_20offset_230to500words.npz')
train_D_20samplesPerWord_20offset_230to500words = train_DSY_20samplesPerWord_20offset_230to500words['trainDense']
train_S_20samplesPerWord_20offset_230to500words = train_DSY_20samplesPerWord_20offset_230to500words['trainSoftmax']

train_DSY_20samplesPerWord_40offset_230to500words = np.load('LRW_train_dense_softmax_y_20samplesPerWord_40offset_230to500words.npz')
train_D_20samplesPerWord_40offset_230to500words = train_DSY_20samplesPerWord_40offset_230to500words['trainDense']
train_S_20samplesPerWord_40offset_230to500words = train_DSY_20samplesPerWord_40offset_230to500words['trainSoftmax']

for w in range(230, 500):
    train_D_50samplesPerWord[(w*50+0):(w*50+20)] = train_D_20samplesPerWord_0offset_230to500words[(w-230)*20:(w-230+1)*20]
    train_D_50samplesPerWord[(w*50+20):(w*50+40)] = train_D_20samplesPerWord_20offset_230to500words[(w-230)*20:(w-230+1)*20]
    train_D_50samplesPerWord[(w*50+40):(w*50+50)] = train_D_20samplesPerWord_40offset_230to500words[(w-230)*20:((w-230)*20+10)]
    train_S_50samplesPerWord[(w*50+0):(w*50+20)] = train_S_20samplesPerWord_0offset_230to500words[(w-230)*20:(w-230+1)*20]
    train_S_50samplesPerWord[(w*50+20):(w*50+40)] = train_S_20samplesPerWord_20offset_230to500words[(w-230)*20:(w-230+1)*20]
    train_S_50samplesPerWord[(w*50+40):(w*50+50)] = train_S_20samplesPerWord_40offset_230to500words[(w-230)*20:((w-230)*20+10)]
    train_Y_50samplesPerWord[w*50:(w+1)*50] = correct_lrw_softmax_argmax[w]

np.savez('LRW_train_dense_softmax_y_50samplesPerWord',
         lrw_train_dense=train_D_50samplesPerWord,
         lrw_train_softmax=train_S_50samplesPerWord,
         lrw_correct_one_hot_arg=train_Y_50samplesPerWord)


