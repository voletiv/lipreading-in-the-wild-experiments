import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

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
# ACCURACIES
#############################################################

np.sum(lrw_lipreader_preds_val_wordIdx == LRW_CORRECT_WORDIDX)/len(LRW_CORRECT_WORDIDX)
np.sum(lrw_lipreader_preds_test_wordIdx == LRW_CORRECT_WORDIDX)/len(LRW_CORRECT_WORDIDX)

# for i in range(0, 25000, 50):
#     print(np.sum(lrw_lipreader_preds_val_wordIdx[i:i+50]==np.argmax(np.bincount(lrw_lipreader_preds_val_wordIdx[i:i+50])))/50,
#         np.sum(lrw_lipreader_preds_test_wordIdx[i:i+50]==np.argmax(np.bincount(lrw_lipreader_preds_test_wordIdx[i:i+50])))/50)

#############################################################
# PRECISION-AT-K
#############################################################

n_classes = LRW_VOCAB_SIZE
samples_per_class = LRW_TEST_SAMPLES_PER_CLASS

lrw_val_precision_at_k_per_word = find_precision_at_k_and_average_precision(lrw_lipreader_preds_val_softmax, LRW_CORRECT_WORDIDX)

lrw_test_precision_at_k_per_word = find_precision_at_k_and_average_precision(lrw_lipreader_preds_test_softmax, LRW_CORRECT_WORDIDX)

np.savez("lrw_precision_at_k",
    # lrw_val_average_precision_per_word=lrw_val_average_precision_per_word,
    lrw_val_precision_at_k_per_word=lrw_val_precision_at_k_per_word,
    # lrw_test_average_precision_per_word=lrw_test_average_precision_per_word,
    lrw_test_precision_at_k_per_word=lrw_test_precision_at_k_per_word)


# # BAR GRAPH OF AVERAGE PRECISION PER WORD
# plt.barh(np.arange(n_classes), val_precision_at_k_per_word[-1])
# plt.yticks(np.arange(n_classes), LRW_VOCAB, fontsize=8)
# plt.show()

plot_lrw_property_image(lrw_val_precision_at_k_per_word[-1], title="Average Precision (@K) - LRW val", cmap='gray', clim=[0, 1])

plot_lrw_property_image(lrw_test_precision_at_k_per_word[-1], title="Average Precision (@K) - LRW test", cmap='jet', clim=[0, 1])





#############################################################
# CONFUSION MATRIX
#############################################################

# train_critic_tn, train_critic_fp, train_critic_fn, train_critic_tp = confusion_matrix(LRW_CORRECT_WORDIDX, train_critic_preds > .5).ravel()
# train_critic_OP_fpr = train_critic_fp/(train_critic_fp + train_critic_tn)
# train_critic_OP_tpr = train_critic_tp/(train_critic_tp + train_critic_fn)

val_fpr, val_tpr, val_roc_auc, val_OP_fpr, val_OP_tpr, \
        test_fpr, test_tpr, test_roc_auc, test_OP_fpr, test_OP_tpr = \
    compute_ROC_lrw_multiclass(LRW_CORRECT_WORDIDX,
        lrw_lipreader_preds_val_softmax,
        lrw_lipreader_preds_test_softmax,
        savePlot=False, showPlot=True,
        plot_title='ROC curve of LRW lipreader (AJ)')

#############################################################
# PR CURVE
#############################################################

n_classes = 500
Y_test = label_binarize(LRW_CORRECT_WORDIDX, classes=range(n_classes))
y_score = lrw_lipreader_preds_val_softmax

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

plt.figure()
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
         where='post')
plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))

# Average precision score, micro-averaged over all classes: 0.79


recall_OP, precision_OP, precision, recall, average_precision \
    = compute_grid_multiclass_PR_plot_curve(LRW_CORRECT_WORDIDX, lrw_lipreader_preds_val_softmax, lrw_val_preds_word_idx, plotCurve=False)


recall_OP, precision_OP, precision, recall, average_precision \
    = compute_grid_multiclass_PR_plot_curve(LRW_CORRECT_WORDIDX, lrw_lipreader_preds_test_softmax, lrw_test_preds_word_idx, plotCurve=False)










#############################################################
# DUMMY
#############################################################

correct = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 2])
correct_binarized = label_binarize(correct, classes=np.arange(3))

ys = np.reshape([.9, .05, .05]*len(correct), (len(correct), 3))

ts = np.reshape([.05, .9, .05]*len(correct), (len(correct), 3))

np.sum(np.argmax(ys, axis=1) == correct)/len(correct)

np.mean(np.sum(label_binarize(np.argmax(ys, axis=1), classes=np.arange(3)) == label_binarize(correct, classes=np.arange(3)), axis=0)/len(correct))

compute_ROC_lrw_multiclass(correct,
        ys,
        ts,
        savePlot=False, showPlot=True,
        plot_title='ROC curve of dummy')


# POS - lipreader classified correctly
# TP - # of images classified correctly
# If 1 image is classified wrongly, TP = total - 1, FP = 1
