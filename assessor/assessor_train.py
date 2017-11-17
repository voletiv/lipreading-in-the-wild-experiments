import os

from assessor_functions import *
from assessor_model import *
from assessor_train_params import *

######################################################
# GEN BATCHES OF IMAGES
######################################################

train_generator = generate_assessor_data_batches(data_dir=data_dir, batch_size=batch_size, collect_type=train_collect_type, shuffle=shuffle, random_crop=random_crop, verbose=verbose)

val_generator = generate_assessor_data_batches(data_dir=data_dir, batch_size=batch_size, collect_type=val_collect_type, shuffle=True, random_crop=False, verbose=False)

######################################################
# MAKE MODEL
######################################################

assessor = my_assessor_model(mouth_nn, mouth_features_dim, lstm_units_1, dense_fc_1, dense_fc_2)

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
