import os

from assessor_functions import *
from assessor_model import *
from assessor_train_params import *


######################################################
# DIR, PARAMS
######################################################

# Make the dir if it doesn't exist
if not os.path.exists(this_assessor_save_dir):
    print("Making dir", this_assessor_save_dir)
    os.makedirs(this_assessor_save_dir)

# Copy train_params file into this_assessor_save_dir
os.system("cp assessor_train_params.py " + this_assessor_save_dir)
print("Copied assessor_train_params.py to", this_assessor_save_dir)


######################################################
# GEN BATCHES OF IMAGES
######################################################

train_generator = generate_assessor_data_batches(data_dir=data_dir, batch_size=batch_size, collect_type=train_collect_type, shuffle=shuffle, random_crop=random_crop, verbose=verbose)

val_generator = generate_assessor_data_batches(data_dir=data_dir, batch_size=batch_size, collect_type=val_collect_type, shuffle=True, random_crop=False, verbose=False)

######################################################
# MAKE MODEL
######################################################

assessor = my_assessor_model(mouth_nn, mouth_features_dim, lstm_units_1, dense_fc_1, dense_fc_2, conv_f_1=conv_f_1, conv_f_2=conv_f_2, conv_f_3=conv_f_3)

assessor.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

write_model_architecture(assessor, file_type='json', file_name=os.path.join(this_assessor_save_dir, this_assessor_model))
write_model_architecture(assessor, file_type='yaml', file_name=os.path.join(this_assessor_save_dir, this_assessor_model))

######################################################
# CALLBACKS
######################################################

checkpointAndMakePlots = CheckpointAndMakePlots(file_name_pre=this_assessor_model, this_assessor_save_dir=this_assessor_save_dir)

######################################################
# TRAIN
######################################################

try:
    assessor.fit_generator(train_generator,
                           steps_per_epoch=train_steps_per_epoch,
                           # steps_per_epoch=1,
                           epochs=n_epochs,
                           callbacks=[checkpointAndMakePlots],
                           validation_data=val_generator,
                           validation_steps=val_steps_per_epoch,
                           # validation_steps=1,
                           class_weight=class_weight,
                           initial_epoch=0)

except KeyboardInterrupt:
    print("Saving latest weights as", os.path.join(this_assessor_save_dir, this_assessor_model+"_assessor.hdf5"), "...")
    assessor.save_weights(os.path.join(this_assessor_save_dir, this_assessor_model+"_assessor.hdf5"))
    print("Done.")
