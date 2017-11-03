import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.externals import joblib

from lrw_image_retrieval_functions import *

#############################################################
# Basics
#############################################################

LRW_VOCAB = load_vocab(LRW_VOCAB_FILE, sort=True)
LRW_CORRECT_WORDIDX = np.repeat(np.arange(500), 50)

# When ordered according to magnetar_LRW_all_words.txt, i.e. not alphabetically:
# LRW_VOCAB = load_vocab(LRW_VOCAB_FILE, sort=False)
# LRW_CORRECT_WORDIDX = load_lrw_correct_wordIdx()

#############################################################
# LOAD DENSE, SOFTMAX OUTPUTS
#############################################################

lrw_lipreader_preds_val_dense, lrw_lipreader_preds_val_softmax, \
        lrw_lipreader_preds_test_dense, lrw_lipreader_preds_test_softmax = \
    load_lrw_dense_softmax_from_mat_file(LRW_LIPREADER_OUTPUTS_MAT_FILE)

#############################################################
# FIX ORDER
#############################################################

a = {'lrw_lipreader_preds_val_dense':lrw_lipreader_preds_val_dense,
     'lrw_lipreader_preds_val_softmax':lrw_lipreader_preds_val_softmax,
     'lrw_lipreader_preds_test_dense':lrw_lipreader_preds_test_dense,
     'lrw_lipreader_preds_test_softmax':lrw_lipreader_preds_test_softmax}

fix_order_of_features_and_samples(a)

lrw_lipreader_preds_val_dense = a['lrw_lipreader_preds_val_dense']
lrw_lipreader_preds_val_softmax = a['lrw_lipreader_preds_val_softmax']
lrw_lipreader_preds_test_dense = a['lrw_lipreader_preds_test_dense']
lrw_lipreader_preds_test_softmax = a['lrw_lipreader_preds_test_softmax']

#############################################################
# LRW_LIPREADER_PREDS_WORDIDX
#############################################################

lrw_lipreader_preds_val_wordIdx = np.argmax(lrw_lipreader_preds_val_softmax, axis=1)
lrw_lipreader_preds_test_wordIdx = np.argmax(lrw_lipreader_preds_test_softmax, axis=1)

#############################################################
# LRW_LIPREADER_PREDS_CORRECT_OR_WRONG
#############################################################

lrw_lipreader_preds_val_correct_or_wrong = lrw_lipreader_preds_val_wordIdx == LRW_CORRECT_WORDIDX
lrw_lipreader_preds_test_correct_or_wrong = lrw_lipreader_preds_test_wordIdx == LRW_CORRECT_WORDIDX

lrw_val_accuracy_per_word = [np.mean(lrw_lipreader_preds_val_correct_or_wrong[i:i+50]) for i in range(0, 25000, 50)]
lrw_test_accuracy_per_word = [np.mean(lrw_lipreader_preds_test_correct_or_wrong[i:i+50]) for i in range(0, 25000, 50)]

#############################################################
# LRW_WORD_DURATIONS
#############################################################

# lrw_word_mean_durations, _ = blazar_LRW_word_durations()
lrw_word_mean_durations = load_lrw_words_mean_durations()

# plot_lrw_property_image(blazar_LRW_word_mean_durations, title="word_durations", cmap='gray')
plot_lrw_property_image(lrw_word_mean_durations, title="lrw_word_durations", cmap='gray')

lrw_word_mean_durations_per_sample = np.repeat(lrw_word_mean_durations, (LRW_TEST_SAMPLES_PER_CLASS))

#############################################################
# ATTRIBUTES
#############################################################

lrw_val_attributes = np.empty((len(lrw_word_mean_durations_per_sample), 0))

lrw_val_attributes = np.hstack((lrw_val_attributes, np.reshape(lrw_word_mean_durations_per_sample,
                                                               (len(lrw_word_mean_durations_per_sample), 1)),
                                lrw_lipreader_preds_val_dense,
                                lrw_lipreader_preds_val_softmax))

lrw_test_attributes = np.empty((len(lrw_word_mean_durations_per_sample), 0))

lrw_test_attributes = np.hstack((lrw_test_attributes, np.reshape(lrw_word_mean_durations_per_sample,
                                                               (len(lrw_word_mean_durations_per_sample), 1)),
                                lrw_lipreader_preds_test_dense,
                                lrw_lipreader_preds_test_softmax))


#############################################################
# LOGISTIC REGRESSOR CRITIC
#############################################################

logReg_unopt = LogisticRegression()
logReg_unopt.fit(lrw_val_attributes, lrw_lipreader_preds_val_correct_or_wrong)

# Save
joblib.dump(logReg_unopt, 'logReg_unopt_attributes.pkl', compress=3)

# Acc
logReg_unopt.score(lrw_val_attributes, lrw_lipreader_preds_val_correct_or_wrong)
logReg_unopt.score(lrw_test_attributes, lrw_lipreader_preds_test_correct_or_wrong)
# >>> # Acc
# ... logReg_unopt.score(lrw_val_attributes, lrw_lipreader_preds_val_correct_or_wrong)
# 0.73763999999999996
# >>> logReg_unopt.score(lrw_test_attributes, lrw_lipreader_preds_test_correct_or_wrong)
# 0.71192

# Scores
lrw_val_logReg_unopt_score = logReg_unopt.decision_function(lrw_val_attributes)
lrw_test_logReg_unopt_score = logReg_unopt.decision_function(lrw_test_attributes)
lrw_val_logReg_unopt_prob = logReg_unopt.predict_proba(lrw_val_attributes)[:, 1]
lrw_test_logReg_unopt_prob = logReg_unopt.predict_proba(lrw_test_attributes)[:, 1]

# OP
critic_threshold = 0.7
# val
lrw_val_tn, lrw_val_fp, lrw_val_fn, lrw_val_tp = confusion_matrix(lrw_lipreader_preds_val_correct_or_wrong, lrw_val_logReg_unopt_prob >= threshold).ravel()
lrw_val_fpr_op = lrw_val_fp/(lrw_val_fp + lrw_val_tn)
lrw_val_tpr_op = lrw_val_tp/(lrw_val_tp + lrw_val_fn)
lrw_val_acc = np.mean((lrw_val_logReg_unopt_prob[:, 1] >= threshold) == lrw_lipreader_preds_val_correct_or_wrong)
# test
lrw_test_tn, lrw_test_fp, lrw_test_fn, lrw_test_tp = confusion_matrix(lrw_lipreader_preds_test_correct_or_wrong, lrw_test_logReg_unopt_prob >= threshold).ravel()
lrw_test_fpr_op = lrw_test_fp/(lrw_test_fp + lrw_test_tn)
lrw_test_tpr_op = lrw_test_tp/(lrw_test_tp + lrw_test_fn)
lrw_test_acc = np.mean((lrw_test_logReg_unopt_prob[:, 1] >= threshold) == lrw_lipreader_preds_test_correct_or_wrong)

# ROC
# val
lrw_val_fpr, lrw_val_tpr, _ = roc_curve(lrw_lipreader_preds_val_correct_or_wrong, lrw_val_logReg_unopt_score)
lrw_val_roc_auc = auc(lrw_val_fpr, lrw_val_tpr)
plt.plot(lrw_val_fpr, lrw_val_tpr, label='val ROC curve (area = {0:0.2f}), threshold = {1:.02f}, val_acc = {2:.02f}'.format(lrw_val_roc_auc, threshold, lrw_val_acc), color='C0')
# plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot(lrw_val_fpr_op, lrw_val_tpr_op, marker='x', markersize=10, color='C0')
# test
lrw_test_fpr, lrw_test_tpr, _ = roc_curve(lrw_lipreader_preds_test_correct_or_wrong, lrw_test_logReg_unopt_score)
lrw_test_roc_auc = auc(lrw_test_fpr, lrw_test_tpr)
plt.plot(lrw_test_fpr, lrw_test_tpr, label='test ROC curve (area = {0:0.2f}), threshold = {1:.02f}, test_acc = {1:.02f}'.format(lrw_test_roc_auc, threshold, lrw_test_acc), color='C1')
plt.plot(lrw_test_fpr_op, lrw_test_tpr_op, marker='x', markersize=10, color='C1')
plt.title("logReg critic on LRW - using word_durations, dense_preds, softmax_preds")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()

#############################################################
# LOGISTIC REGRESSOR CRITIC - PRECISION @ K
# Reject those lipreader preds that critic says are definitely wrong
#############################################################

critic_threshold = 0.7  # Optimal threshold for OP close to (0, 1)

# BASELINE PRECISIONS @ K
# Val
lrw_val_precision_at_k_per_word = find_precision_at_k_and_average_precision(lrw_lipreader_preds_val_softmax, LRW_CORRECT_WORDIDX, critic_removes=None)
np.mean(lrw_val_precision_at_k_per_word[0])
# 0.99599999999999889
np.mean(lrw_val_precision_at_k_per_word[9])
# 0.98486019620811271
np.mean(lrw_val_precision_at_k_per_word[-1])
# 0.77693021007376872
# Test
lrw_test_precision_at_k_per_word = find_precision_at_k_and_average_precision(lrw_lipreader_preds_test_softmax, LRW_CORRECT_WORDIDX, critic_removes=None)
np.mean(lrw_test_precision_at_k_per_word[0])
# 0.99399999999999888
np.mean(lrw_test_precision_at_k_per_word[9])
# 0.98131201499118148
np.mean(lrw_test_precision_at_k_per_word[-1])
# 0.76906032514427969

# REJECT
lrw_val_rejection_idx = lrw_val_logReg_unopt_prob <= critic_threshold_for_wrong
np.mean(lrw_val_rejection_idx)
# 0.4214
lrw_test_rejection_idx = lrw_test_logReg_unopt_prob <= critic_threshold_for_wrong
np.mean(lrw_test_rejection_idx)
# 0.41927999999999999

# CHECK PRECISIONS @ K
# Val
filtered_lrw_val_precision_at_k_per_word = find_precision_at_k_and_average_precision(lrw_lipreader_preds_val_softmax, LRW_CORRECT_WORDIDX, critic_removes=lrw_val_rejection_idx)
np.mean(filtered_lrw_val_precision_at_k_per_word[0])
# 0.97999999999999887
np.mean(filtered_lrw_val_precision_at_k_per_word[9])
# 0.95923937610229271
np.mean(filtered_lrw_val_precision_at_k_per_word[-1])
# 0.84051497409931819
# Test
filtered_lrw_test_precision_at_k_per_word = find_precision_at_k_and_average_precision(lrw_lipreader_preds_test_softmax, LRW_CORRECT_WORDIDX, critic_removes=lrw_test_rejection_idx)
np.mean(filtered_lrw_val_precision_at_k_per_word[0])
# 0.97999999999999887
np.mean(filtered_lrw_val_precision_at_k_per_word[9])
# 0.95923937610229271
np.mean(filtered_lrw_test_precision_at_k_per_word[-1])
# 0.80930961564990167


# PLOT PRECISIONS @ K, averaged across words

plt.plot(np.mean(lrw_val_precision_at_k_per_word, axis=1), label="P@K of lipreader")
plt.plot(np.mean(filtered_lrw_val_precision_at_k_per_word, axis=1), label="P@K of lipreader minus critic_rejects")
plt.title("LRW Val - mean Average Precisions @ K (mean across words) \n - comparison with rejects by logReg critic")
plt.xlabel("K = # of documents (ranked by respective scores)")
plt.ylabel("average precision @ K")
plt.legend()
plt.show()

plt.plot(np.mean(lrw_test_precision_at_k_per_word, axis=1), label="P@K of lipreader")
plt.plot(np.mean(filtered_lrw_test_precision_at_k_per_word, axis=1), label="P@K of lipreader minus critic_rejects")
plt.title("LRW Test - mean Average Precisions @ K (mean across words) \n - comparison with rejects by logReg critic")
plt.xlabel("K = # of documents (ranked by respective scores)")
plt.ylabel("average precision @ K")
plt.legend()
plt.show()


# PLOT AVERAGE PRECISION @ K vs WORDS
plot_lrw_property_image(filtered_lrw_val_precision_at_k_per_word[-1], title="Average Precision (@K) - LRW_val minus critic_rejects", cmap='gray', clim=[0, 1])

plot_lrw_property_image(filtered_lrw_test_precision_at_k_per_word[-1], title="Average Precision (@K) - LRW_test minus critic_rejects", cmap='gray', clim=[0, 1])


