from sklearn.metrics import confusion_matrix, roc_curve, auc, average_precision_score, precision_recall_curve

from assessor_evaluation_functions import *
from assessor_train_params import *

######################################################
# PREDICT
######################################################

# assessor_preds = assessor.predict_generator(train_generator, train_steps_per_epoch)

batch_size = 100

# LRW_VAL
lrw_val_generator = generate_assessor_training_batches(data_dir=data_dir, batch_size=batch_size, collect_type="val", shuffle=False, random_crop=False, verbose=verbose)

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
    [X, y] = next(train_generator)
    lrw_val_assessor_preds = np.append(lrw_val_assessor_preds, assessor.predict(X))

# Save
np.save(os.path.join(assessor_save_dir, this_model+"_lrw_val_preds"), lrw_val_assessor_preds)

# LRW_TEST
lrw_test_generator = generate_assessor_training_batches(data_dir=data_dir, batch_size=batch_size, collect_type="test", shuffle=False, random_crop=False, verbose=False)

lrw_test_assessor_preds = np.array([])
for i in tqdm.tqdm(range(25000//batch_size)):
    [X, y] = next(lrw_test_generator)
    lrw_test_assessor_preds = np.append(lrw_test_assessor_preds, assessor.predict(X))

# Save
np.save(os.path.join(assessor_save_dir, this_model+"_lrw_test_preds"), lrw_test_assessor_preds)

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

assessor_threshold = .7

######################################################
# ROC, OPERATING POINT
######################################################

# OP
tn, fp, fn, tp = confusion_matrix(lipreader_lrw_val_correct_or_wrong, lrw_val_assessor_preds >= assessor_threshold).ravel()
fpr_op = fp/(fp + tn)
tpr_op = tp/(tp + fn)

# ROC
fpr, tpr, thresholds = roc_curve(lipreader_lrw_val_correct_or_wrong, lrw_val_assessor_preds)
roc_auc = auc(fpr, tpr)

# Plot
plot_ROC_with_OP(fpr, tpr, roc_auc, fpr_op, tpr_op, assessor_save_dir, this_model, assessor_threshold)

######################################################
# P-R CURVE
######################################################

# PR
average_precision = average_precision_score(lipreader_lrw_val_correct_or_wrong, lrw_val_assessor_preds)
precision, recall, _ = precision_recall_curve(lipreader_lrw_val_correct_or_wrong, lrw_val_assessor_preds)

plot_PR_curve(recall, precision, average_precision, assessor_save_dir, this_model)


######################################################
# COMPARISON OF P-R
######################################################

lipreader_lrw_val_precision_w, lipreader_lrw_val_recall_w = my_precision_recall(lipreader_lrw_val_softmax, lrw_correct_one_hot_y_arg)

lipreader_lrw_val_precision_at_k_averaged_across_words = np.mean(lipreader_lrw_val_precision_w, axis=1)[:50]
lipreader_lrw_val_recall_at_k_averaged_across_words = np.mean(lipreader_lrw_val_recall_w, axis=1)[:50]

lrw_val_rejection_idx = lrw_val_assessor_preds <= assessor_threshold
filtered_lipreader_lrw_val_precision_w, filtered_lipreader_lrw_val_val_recall_w = my_precision_recall(lipreader_lrw_val_softmax, lrw_correct_one_hot_y_arg, critic_removes=lrw_val_rejection_idx)

filtered_val_precision_at_k_averaged_across_words = np.mean(filtered_lipreader_lrw_val_precision_w, axis=1)[:50]
filtered_val_recall_at_k_averaged_across_words = np.mean(filtered_lipreader_lrw_val_val_recall_w, axis=1)[:50]

# P@K vs K, R@K vs K
plot_P_atK_and_R_atK_vs_K(lipreader_lrw_val_precision_at_k_averaged_across_words, filtered_val_precision_at_k_averaged_across_words,
                          lipreader_lrw_val_recall_at_k_averaged_across_words, filtered_val_recall_at_k_averaged_across_words,
                          assessor_save_dir=assessor_save_dir, this_model="assessor_cnn_adam", lrw_type="val"+str(assessor_threshold))

# P-R curve VAL
plot_P_atK_vs_R_atK(lipreader_lrw_val_precision_at_k_averaged_across_words, filtered_val_precision_at_k_averaged_across_words,
                    lipreader_lrw_val_recall_at_k_averaged_across_words, filtered_val_recall_at_k_averaged_across_words,
                    assessor_save_dir=assessor_save_dir, this_model="assessor_cnn_adam", lrw_type="val"+str(assessor_threshold))




