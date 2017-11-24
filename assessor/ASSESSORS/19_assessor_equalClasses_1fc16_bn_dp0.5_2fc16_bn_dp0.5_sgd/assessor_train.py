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

# Copy assessor_model file into this_assessor_save_dir
os.system("cp assessor_model.py " + this_assessor_save_dir)
print("Copied assessor_model.py to", this_assessor_save_dir)

# Copy assessor_train_params file into this_assessor_save_dir
os.system("cp assessor_train_params.py " + this_assessor_save_dir)
print("Copied assessor_train_params.py to", this_assessor_save_dir)

# Copy assessor_train file into this_assessor_save_dir
os.system("cp assessor_train.py " + this_assessor_save_dir)
print("Copied assessor_train.py to", this_assessor_save_dir)

######################################################
# GEN BATCHES OF IMAGES
######################################################

train_generator = generate_assessor_data_batches(batch_size=batch_size, data_dir=data_dir, collect_type=train_collect_type,
                                                 shuffle=shuffle, equal_classes=equal_classes, use_CNN_LSTM=use_CNN_LSTM,
                                                 grayscale_images=grayscale_images, random_crop=random_crop, random_flip=random_flip, verbose=verbose)

val_generator = generate_assessor_data_batches(batch_size=batch_size, data_dir=data_dir, collect_type=val_collect_type,
                                               shuffle=shuffle, equal_classes=equal_classes, use_CNN_LSTM=use_CNN_LSTM,
                                               grayscale_images=grayscale_images, random_crop=False, random_flip=False, verbose=verbose)

######################################################
# MAKE MODEL
######################################################

assessor = my_assessor_model(use_CNN_LSTM=use_CNN_LSTM, mouth_nn=mouth_nn,
                             conv_f_1=conv_f_1, conv_f_2=conv_f_2, conv_f_3=conv_f_3,
                             mouth_features_dim=mouth_features_dim, lstm_units_1=lstm_units_1,
                             dense_fc_1=dense_fc_1, dense_fc_2=dense_fc_2,
                             grayscale_images=grayscale_images)

assessor.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

write_model_architecture(assessor, file_type='json', file_name=os.path.join(this_assessor_save_dir, this_assessor_model))
write_model_architecture(assessor, file_type='yaml', file_name=os.path.join(this_assessor_save_dir, this_assessor_model))

######################################################
# CALLBACKS
######################################################

# lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=0.5e-6)

early_stopper = EarlyStopping(min_delta=0.001, patience=50)

checkpointAndMakePlots = CheckpointAndMakePlots(file_name_pre=this_assessor_model, this_assessor_save_dir=this_assessor_save_dir)

######################################################
# TRAIN
######################################################

saved_final_model = False

try:
    assessor.fit_generator(train_generator,
                           steps_per_epoch=train_steps_per_epoch,
                           # steps_per_epoch=1,
                           epochs=n_epochs,
                           # callbacks=[lr_reducer, early_stopper, checkpointAndMakePlots],
                           callbacks=[early_stopper, checkpointAndMakePlots],
                           validation_data=val_generator,
                           validation_steps=val_steps_per_epoch,
                           # validation_steps=1,
                           class_weight=class_weight,
                           initial_epoch=0)

except KeyboardInterrupt:
    print("Saving latest weights as", os.path.join(this_assessor_save_dir, this_assessor_model+"_assessor.hdf5"), "...")
    assessor.save_weights(os.path.join(this_assessor_save_dir, this_assessor_model+"_assessor.hdf5"))
    saved_final_model = True

if not saved_final_model:
    print("Saving latest weights as", os.path.join(this_assessor_save_dir, this_assessor_model+"_assessor.hdf5"), "...")
    assessor.save_weights(os.path.join(this_assessor_save_dir, this_assessor_model+"_assessor.hdf5"))
    saved_final_model = True

print("Done.")
