import glob
import os
import tqdm

from keras.models import load_model

from assessor_functions import *

#####################################
# ASSESSORS
#####################################

assessor_experiment_numbers = [33, 36, 37, 39, 40, 42, 43]

assessor_dirs = [sorted(glob.glob(os.path.join(ASSESSOR_SAVE_DIR, str(e)+"_*/")))[-1] for e in assessor_experiment_numbers]

#####################################
# ASSESSORS - EVALUATE
#####################################

_, lipreader_lrw_val_softmax, lrw_correct_one_hot_y_arg = load_dense_softmax_y(collect_type="val")
_, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg = load_dense_softmax_y(collect_type="test")

lrw_val_assessor_preds = np.empty((0, 25000))
lrw_test_assessor_preds = np.empty((0, 25000))

for assessor_dir in tqdm.tqdm(assessor_dirs):
    print("")
    print("EVALUATING", os.path.normpath(assessor_dir).split('/')[-1], "...")
    assessor_preds = np.load(os.path.join(assessor_dir, "assessor_preds.npz"))
    lrw_val_assessor_preds = np.vstack((lrw_val_assessor_preds, assessor_preds["lrw_val_assessor_preds"]))
    lrw_test_assessor_preds = np.vstack((lrw_test_assessor_preds, assessor_preds["lrw_test_assessor_preds"]))
    evaluate_assessor(lrw_val_assessor_preds=assessor_preds["lrw_val_assessor_preds"],
                      lrw_test_assessor_preds=assessor_preds["lrw_test_assessor_preds"],
                      assessor="blah",
                      assessor_save_dir=assessor_dir,
                      assessor_threshold=0.5,
                      lipreader_lrw_val_softmax=lipreader_lrw_val_softmax,
                      lipreader_lrw_test_softmax=lipreader_lrw_test_softmax,
                      lrw_correct_one_hot_y_arg=lrw_correct_one_hot_y_arg)

#####################################
# ENSEMBLE ASSESSOR - EVALUATE
#####################################

ensemble_lrw_val_assessor_preds = np.mean(lrw_val_assessor_preds, axis=0)
ensemble_lrw_test_assessor_preds = np.mean(lrw_test_assessor_preds, axis=0)

ensemble_assessor_save_dir = os.path.join(ASSESSOR_SAVE_DIR, "CNN_ENSEMBLE_ASSESSOR")

if not os.path.exists(ensemble_assessor_save_dir):
    print("Making dir", ensemble_assessor_save_dir)
    os.makedirs(ensemble_assessor_save_dir)

np.savez(os.path.join(ensemble_assessor_save_dir, "assessor_preds"),
         ensemble_lrw_val_assessor_preds=ensemble_lrw_val_assessor_preds,
         ensemble_lrw_test_assessor_preds=ensemble_lrw_test_assessor_preds)

evaluate_assessor(lrw_val_assessor_preds=ensemble_lrw_val_assessor_preds,
                  lrw_test_assessor_preds=ensemble_lrw_test_assessor_preds,
                  assessor="blah",
                  assessor_save_dir=ensemble_assessor_save_dir,
                  assessor_threshold=0.5,
                  lipreader_lrw_val_softmax=lipreader_lrw_val_softmax,
                  lipreader_lrw_test_softmax=lipreader_lrw_test_softmax,
                  lrw_correct_one_hot_y_arg=lrw_correct_one_hot_y_arg)


#####################################
# ASSESSORS - PREDICT
#####################################

assessors = []

# Load models
for assessor_dir in tqdm.tqdm(assessor_dirs):
    assessors.append(load_model(os.path.join(assessor_dir, "assessor.hdf5")))

# Compile them
for assessor in tqdm.tqdm(assessors):
    assessor.compile(optimizer='adam', loss='binary_crossentropy')

# Predict val
eval_batch_size = 100
lrw_val_data_generator = generate_assessor_data_batches(batch_size=eval_batch_size, data_dir=LRW_DATA_DIR, collect_type="val", shuffle=False, equal_classes=False,
                                                        use_CNN_LSTM=True, mouth_nn="syncnet_preds", use_head_pose=True, use_softmax=True,
                                                        grayscale_images=True, random_crop=False, random_flip=False, verbose=False)

lrw_test_data_generator = generate_assessor_data_batches(batch_size=eval_batch_size, data_dir=LRW_DATA_DIR, collect_type="test", shuffle=False, equal_classes=False,
                                                         use_CNN_LSTM=True, mouth_nn="syncnet_preds", use_head_pose=True, use_softmax=True,
                                                         grayscale_images=True, random_crop=False, random_flip=False, verbose=False)

for i in tqdm.tqdm(range(len(assessor_experiment_numbers))):
    assessor_dir = assessor_dirs[i]
    assessor = assessors[i]
    lrw_val_assessor_preds = np.array([])
    lrw_test_assessor_preds = np.array([])
    for batch in tqdm.tqdm(range(25000//eval_batch_size)):
        (X, _) = next(lrw_val_data_generator)
        lrw_val_assessor_preds = np.append(lrw_val_assessor_preds, assessor.predict(X))
    for batch in tqdm.tqdm(range(25000//eval_batch_size)):
        (X, _) = next(lrw_test_data_generator)
        lrw_test_assessor_preds = np.append(lrw_test_assessor_preds, assessor.predict(X))
    np.savez(os.path.join(assessor_dir, "assessor_preds"), lrw_val_assessor_preds=lrw_val_assessor_preds, lrw_test_assessor_preds=lrw_test_assessor_preds)
