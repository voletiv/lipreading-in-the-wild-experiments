import glob

from sklearn.metrics import confusion_matrix, roc_curve, auc, average_precision_score, precision_recall_curve

from assessor_evaluation_functions import *
from assessor_model import *
from assessor_train_params import *

######################################################
# SAVE MODEL
######################################################

# assessor.save_weights("assessor.hdf5")

######################################################
# LOAD MODEL
######################################################

experiment_number = 3

for save_dir in sorted(glob.glob(os.path.join(ASSESSOR_SAVE_DIR, "*/"))):
    if int(save_dir.split('/')[-2][0]) == experiment_number:
        assessor_save_dir = save_dir
        this_model = save_dir.split('/')[-2]
        break

assessor = read_my_model(model_file_name=os.path.join(assessor_save_dir, this_model+".json"),
                         weights_file_name=os.path.join(assessor_save_dir, "3_assessor_cnn_mouth512_lstm32_1fc512_2fc128_adam_epoch004_tl0.2254_ta0.7095_vl0.5716_va0.7038.hdf5"))

######################################################
# PREDICT
######################################################

# assessor_preds = assessor.predict_generator(train_generator, train_steps_per_epoch)

batch_size = 100

# LRW_VAL
lrw_val_generator = generate_assessor_data_batches(data_dir=data_dir, batch_size=batch_size, collect_type="val", shuffle=False, random_crop=False, verbose=verbose)

# # FAST?
# get_output = K.function([assessor.input[0], assessor.input[1], assessor.input[2], assessor.input[3], assessor.input[4], K.learning_phase()],
#                         [assessor.output])
# lrw_val_assessor_preds = np.array([])
# for i in tqdm.tqdm(range(25000//batch_size)):
#     [X, y] = next(train_generator)
#     lrw_val_assessor_preds = np.append(lrw_val_assessor_preds, get_output([X[0], X[1], np.reshape(X[2], (batch_size, 1)), X[3], X[4], 0])[0])

# SLOW?
lrw_val_assessor_preds = np.array([])
for i in tqdm.tqdm(range(25000//batch_size)):
    [X, y] = next(lrw_val_generator)
    lrw_val_assessor_preds = np.append(lrw_val_assessor_preds, assessor.predict(X))

# Save
np.save(os.path.join(assessor_save_dir, this_model+"_lrw_val_preds"), lrw_val_assessor_preds)

# LRW_TEST
lrw_test_generator = generate_assessor_data_batches(data_dir=data_dir, batch_size=batch_size, collect_type="test", shuffle=False, random_crop=False, verbose=False)

lrw_test_assessor_preds = np.array([])
for i in tqdm.tqdm(range(25000//batch_size)):
    [X, y] = next(lrw_test_generator)
    lrw_test_assessor_preds = np.append(lrw_test_assessor_preds, assessor.predict(X))

# Save
np.save(os.path.join(assessor_save_dir, this_model+"_lrw_test_preds"), lrw_test_assessor_preds)

######################################################
# LOAD
######################################################

lrw_val_assessor_preds = np.load(os.path.join(assessor_save_dir, this_model+"_lrw_val_preds.npy"))

lrw_test_assessor_preds = np.load(os.path.join(assessor_save_dir, this_model+"_lrw_test_preds.npy"))

######################################################
# SOFTMAX
######################################################

# LRW_VAL
lipreader_lrw_val_dense, lipreader_lrw_val_softmax, lrw_correct_one_hot_y_arg = load_dense_softmax_y(collect_type="val")
lipreader_lrw_val_correct_or_wrong = np.argmax(lipreader_lrw_val_softmax, axis=1) == lrw_correct_one_hot_y_arg

# LRW_TEST
lipreader_lrw_test_dense, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg = load_dense_softmax_y(collect_type="val")
lipreader_lrw_test_correct_or_wrong = np.argmax(lipreader_lrw_test_softmax, axis=1) == lrw_correct_one_hot_y_arg

######################################################
# ASSESSOR THRESHOLD
######################################################

assessor_threshold = .5

######################################################
# ROC, OPERATING POINT
######################################################

# VAL

# OP
tn, fp, fn, tp = confusion_matrix(lipreader_lrw_val_correct_or_wrong, lrw_val_assessor_preds >= assessor_threshold).ravel()
fpr_op = fp/(fp + tn)
tpr_op = tp/(tp + fn)

# ROC
fpr, tpr, thresholds = roc_curve(lipreader_lrw_val_correct_or_wrong, lrw_val_assessor_preds)
roc_auc = auc(fpr, tpr)

# Plot
plot_ROC_with_OP(fpr, tpr, roc_auc, fpr_op, tpr_op, assessor_save_dir, this_model, lrw_type="val", threshold=assessor_threshold, save_and_close=False)

# TEST

# OP
tn, fp, fn, tp = confusion_matrix(lipreader_lrw_test_correct_or_wrong, lrw_test_assessor_preds >= assessor_threshold).ravel()
fpr_op = fp/(fp + tn)
tpr_op = tp/(tp + fn)

# ROC
fpr, tpr, thresholds = roc_curve(lipreader_lrw_test_correct_or_wrong, lrw_test_assessor_preds)
roc_auc = auc(fpr, tpr)

# Plot
plot_ROC_with_OP(fpr, tpr, roc_auc, fpr_op, tpr_op, assessor_save_dir, this_model, lrw_type="test", threshold=assessor_threshold, save_and_close=True)

######################################################
# P-R CURVE
######################################################

# VAL

# PR
average_precision = average_precision_score(lipreader_lrw_val_correct_or_wrong, lrw_val_assessor_preds)
precision, recall, _ = precision_recall_curve(lipreader_lrw_val_correct_or_wrong, lrw_val_assessor_preds)

plot_assessor_PR_curve(recall, precision, average_precision, assessor_save_dir, this_model, lrw_type="val", save_and_close=False)

# TEST

# PR
average_precision = average_precision_score(lipreader_lrw_test_correct_or_wrong, lrw_test_assessor_preds)
precision, recall, _ = precision_recall_curve(lipreader_lrw_test_correct_or_wrong, lrw_test_assessor_preds)

plot_assessor_PR_curve(recall, precision, average_precision, assessor_save_dir, this_model, lrw_type="test", save_and_close=True)

######################################################
# COMPARISON OF P-R
######################################################

# VAL

lipreader_lrw_val_precision_w, lipreader_lrw_val_recall_w, lipreader_lrw_val_avg_precision_w = \
    my_precision_recall(lipreader_lrw_val_softmax, lrw_correct_one_hot_y_arg)

lipreader_lrw_val_precision_at_k_averaged_across_words = np.mean(lipreader_lrw_val_precision_w, axis=1)[:50]
lipreader_lrw_val_recall_at_k_averaged_across_words = np.mean(lipreader_lrw_val_recall_w, axis=1)[:50]

lrw_val_rejection_idx = lrw_val_assessor_preds <= assessor_threshold
filtered_lipreader_lrw_val_precision_w, filtered_lipreader_lrw_val_recall_w, filtered_lipreader_lrw_val_avg_precision_w = \
    my_precision_recall(lipreader_lrw_val_softmax, lrw_correct_one_hot_y_arg, critic_removes=lrw_val_rejection_idx)

filtered_lrw_val_precision_at_k_averaged_across_words = np.mean(filtered_lipreader_lrw_val_precision_w, axis=1)[:50]
filtered_lrw_val_recall_at_k_averaged_across_words = np.mean(filtered_lipreader_lrw_val_recall_w, axis=1)[:50]

# P@K vs K, R@K vs K
plot_P_atK_and_R_atK_vs_K(lipreader_lrw_val_precision_at_k_averaged_across_words, filtered_lrw_val_precision_at_k_averaged_across_words,
                          lipreader_lrw_val_recall_at_k_averaged_across_words, filtered_lrw_val_recall_at_k_averaged_across_words,
                          assessor_save_dir=assessor_save_dir, this_model=this_model, lrw_type="val", threshold=assessor_threshold)

# P-R curve
plot_P_atK_vs_R_atK(lipreader_lrw_val_precision_at_k_averaged_across_words, filtered_lrw_val_precision_at_k_averaged_across_words,
                    lipreader_lrw_val_recall_at_k_averaged_across_words, filtered_lrw_val_recall_at_k_averaged_across_words,
                    assessor_save_dir=assessor_save_dir, this_model=this_model, lrw_type="val", threshold=assessor_threshold)

# TEST

lipreader_lrw_test_precision_w, lipreader_lrw_test_recall_w, lipreader_lrw_test_avg_precision_w = \
    my_precision_recall(lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg)

lipreader_lrw_test_precision_at_k_averaged_across_words = np.mean(lipreader_lrw_test_precision_w, axis=1)[:50]
lipreader_lrw_test_recall_at_k_averaged_across_words = np.mean(lipreader_lrw_test_recall_w, axis=1)[:50]

lrw_test_rejection_idx = lrw_test_assessor_preds <= assessor_threshold
filtered_lipreader_lrw_test_precision_w, filtered_lipreader_lrw_test_recall_w, filtered_lipreader_lrw_test_avg_precision_w = \
    my_precision_recall(lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg, critic_removes=lrw_test_rejection_idx)

filtered_lrw_test_precision_at_k_averaged_across_words = np.mean(filtered_lipreader_lrw_test_precision_w, axis=1)[:50]
filtered_lrw_test_recall_at_k_averaged_across_words = np.mean(filtered_lipreader_lrw_test_recall_w, axis=1)[:50]

# P@K vs K, R@K vs K
plot_P_atK_and_R_atK_vs_K(lipreader_lrw_test_precision_at_k_averaged_across_words, filtered_lrw_test_precision_at_k_averaged_across_words,
                          lipreader_lrw_test_recall_at_k_averaged_across_words, filtered_lrw_test_recall_at_k_averaged_across_words,
                          assessor_save_dir=assessor_save_dir, this_model=this_model, lrw_type="test", threshold=assessor_threshold)

# P-R curve
plot_P_atK_vs_R_atK(lipreader_lrw_test_precision_at_k_averaged_across_words, filtered_lrw_test_precision_at_k_averaged_across_words,
                    lipreader_lrw_test_recall_at_k_averaged_across_words, filtered_lrw_test_recall_at_k_averaged_across_words,
                    assessor_save_dir=assessor_save_dir, this_model=this_model, lrw_type="test", threshold=assessor_threshold)


######################################################
# PRECISION GIF
######################################################

plot_lrw_property_image(lipreader_lrw_val_avg_precision_w, title="Average Precision (@K) - LRW val", cmap='gray', clim=[0, 1], save=True,
        assessor_save_dir=assessor_save_dir, this_model=this_model, lrw_type="val", file_name="avg_precision"):

plot_lrw_property_image(filtered_lipreader_lrw_val_avg_precision_w, title="Average Precision (@K) - LRW val filtered using assessor", cmap='gray', clim=[0, 1], save=True,
        assessor_save_dir=assessor_save_dir, this_model=this_model, lrw_type="val", file_name="avg_precision"):

