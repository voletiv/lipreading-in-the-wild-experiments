import os

from assessor_functions import *
from assessor_model import *

######################################################
# PARAMS
######################################################

data_dir = LRW_DATA_DIR

batch_size = 32

train_collect_type = "val"

val_collect_type = "test"

shuffle = True

random_crop = True

verbose = False

# Assessor type
mouth_nn = 'cnn'

# Compile
optimizer = 'adam'
loss = 'binary_crossentropy'

# Train
train_lrw_word_set_num_txt_file_names = read_lrw_word_set_num_file_names(collect_type=train_collect_type, collect_by='sample')
train_steps_per_epoch = len(train_lrw_word_set_num_txt_file_names) // batch_size

n_epochs = 100

# Val
val_lrw_word_set_num_txt_file_names = read_lrw_word_set_num_file_names(collect_type=val_collect_type, collect_by='sample')
# val_steps_per_epoch = len(val_lrw_word_set_num_txt_file_names) // batch_size
val_steps_per_epoch = 10     # Set less value so as not to take too much time computing on full val set

# Class weights
# The lipreader is correct 70% of the time
class_weight = {0: .3, 1: .7}

######################################################
# THIS MODEL
######################################################

# THIS MODEL NAME
this_model = "1_assessor_"+mouth_nn+"_"+optimizer

# Save
assessor_save_dir = os.path.realpath(os.path.join('../../ASSESSORS/', this_model))

# Make the dir if it doesn't exist
if not os.path.exists(assessor_save_dir):
    print("Making dir", assessor_save_dir)
    os.makedirs(assessor_save_dir)

######################################################
# GEN BATCHES OF IMAGES
######################################################

train_generator = generate_assessor_training_batches(data_dir=data_dir, batch_size=batch_size, collect_type=train_collect_type, shuffle=shuffle, random_crop=random_crop, verbose=verbose)

val_generator = generate_assessor_training_batches(data_dir=data_dir, batch_size=batch_size, collect_type=val_collect_type, shuffle=True, random_crop=False, verbose=False)

######################################################
# MAKE MODEL
######################################################

assessor = my_assessor_model(mouth_nn)

assessor.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

write_model_architecture(assessor, file_type='json', file_name=os.path.join(assessor_save_dir, this_model))

######################################################
# CALLBACKS
######################################################

checkpointAndMakePlots = CheckpointAndMakePlots(file_name_pre=this_model, assessor_save_dir=assessor_save_dir)

######################################################
# TRAIN
######################################################

history = assessor.fit_generator(train_generator,
                                 steps_per_epoch=train_steps_per_epoch,
                                 epochs=n_epochs,
                                 callbacks=[checkpointAndMakePlots],
                                 validation_data=val_generator,
                                 validation_steps=val_steps_per_epoch,
                                 class_weight=class_weight,
                                 initial_epoch=0)
