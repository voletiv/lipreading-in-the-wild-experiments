import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from scipy import interp
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from lrw_image_retrieval_params import *


#############################################################
# LOAD DENSE, SOFTMAX
#############################################################


def load_lrw_dense_softmax_from_mat_file(mat_file=LRW_LIPREADER_OUTPUTS_MAT_FILE):
    lrw_dense_softmax_mat = sio.loadmat(mat_file)
    lrw_dense_softmax_mat.keys()
    # dict_keys(['test_dense', '__version__', '__globals__', '__header__', 'val_softmax', 'test_softmax', 'val_dense'])
    lrw_val_dense = lrw_dense_softmax_mat['val_dense']          # (25000, 500)
    lrw_val_softmax = lrw_dense_softmax_mat['val_softmax']      # (25000, 500)
    lrw_test_dense = lrw_dense_softmax_mat['test_dense']        # (25000, 500)
    lrw_test_softmax = lrw_dense_softmax_mat['test_softmax']    # (25000, 500)
    return lrw_val_dense, lrw_val_softmax, lrw_test_dense, lrw_test_softmax


#############################################################
# LOAD WORDS VOCABULARY
#############################################################


def load_vocab(vocab_file=LRW_VOCAB_FILE):
    vocab = []
    with open(vocab_file) as f:
        for line in f:
            vocab.append(line.rstrip())
    return vocab


#############################################################
# LOAD WORDS VOCABULARY
#############################################################

LRW_VOCAB = load_vocab(LRW_VOCAB_FILE)

#############################################################
# PRECISION @ K
#############################################################

def find_precision_at_k_and_average_precision(lrw_lipreader_preds_softmax, lrw_correct_wordIdx):
    # Find precision at k for each class
    ranking_sortArgs = np.zeros((lrw_lipreader_preds_softmax.shape), dtype=int)
    ranked_correct_wordIdx = np.zeros((lrw_lipreader_preds_softmax.shape))
    ranked_correct_or_wrong = np.zeros((lrw_lipreader_preds_softmax.shape), dtype=bool)
    precision_at_k_per_word = np.zeros((lrw_lipreader_preds_softmax.shape))
    # For each word
    for c in range(n_classes):
        # Rank the data samples according to confidence
        ranking_sortArgs[:, c] = np.argsort(lrw_lipreader_preds_softmax[:, c])[::-1]
        # Find ranked word idx
        ranked_correct_wordIdx[:, c] = lrw_correct_wordIdx[ranking_sortArgs[:, c]]
        # Check if they are correct or wrong
        ranked_correct_or_wrong[:, c] = (ranked_correct_wordIdx[:, c] == c)
        # Find the precision at each sample
        precision_at_k_per_word[:, c] = np.cumsum( np.cumsum(ranked_correct_or_wrong[:, c]) / np.arange(1, len(ranked_correct_or_wrong[:, c])+1) * ranked_correct_or_wrong[:, c] ) / (1e-15 + np.cumsum(ranked_correct_or_wrong[:, c]))
    # # Precision@1
    # precision_at_1_averaged_over_words = np.mean(precision_at_k_per_word[0])
    # # Precision@10
    # precision_at_10_averaged_over_words = np.mean(precision_at_k_per_word[10])
    # # Precision@50
    # precision_at_50_averaged_over_words = np.mean(precision_at_k_per_word[50])
    # Average Precision
    average_precision_per_word = precision_at_k_per_word[-1]
    average_precision = np.mean(average_precision_per_word)
    # 0.77693021007376883
    return average_precision, precision_at_k_per_word


def plot_lrw_precision_at_k_image(precision_at_k_per_word, title_append="", cmap='jet'):
    # Fig
    fig, ax = plt.subplots(figsize=(28, 10))
    # Grid
    x_lim = 20
    y_lim = 25
    x, y = np.meshgrid(np.arange(x_lim), np.arange(y_lim))
    # Image
    image = plt.imshow(np.reshape(precision_at_k_per_word[-1][:x_lim*y_lim], (y_lim, x_lim)), cmap=cmap, clim=[0, 1], aspect='auto')
    plt.colorbar(image)
    # Words in image
    for i, (x_val, y_val) in enumerate(zip(x.flatten(), y.flatten())):
        # print(image.get_array()[y_val][x_val])
        # if np.mean(image.cmap(image.get_array()[y_val][x_val])[:3]) < .5:
        #     c = 'w'
        # else:
        #     c = 'k'
        # ax.text(x_val, y_val, LRW_VOCAB[i], va='center', ha='center', fontsize=7, color=c)
        ax.text(x_val, y_val, lrw_vocab[i], va='center', ha='center', fontsize=7)
    # ax.set_xlim(0, x_lim)
    # ax.set_ylim(0, y_lim)
    # ax.set_xticks(np.arange(x_lim))
    # ax.set_yticks(np.arange(y_lim))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.grid()
    plt.title("Average Precision (@K) - LRW " + title_append)
    plt.show()
    plt.close()


#############################################################
# ROC COMPUTATION
#############################################################


def compute_ROC_lrw_multiclass(correct_word_idx,
                               val_confidences, test_confidences,
                               savePlot=False, showPlot=False,
                               plot_title='ROC curve'):
    # P => argmax conf is highest confidence
    print("1/{0:01d} Computing val ROC...".format(4 if (savePlot or showPlot) else 2))
    val_fpr, val_tpr, val_roc_auc = \
        compute_ROC_multiclass(label_binarize(correct_word_idx,
                                              classes=np.arange(val_confidences.shape[1])),
                               val_confidences)
    print("2/{0:01d} Computing test ROC...".format(4 if (savePlot or showPlot) else 2))
    test_fpr, test_tpr, test_roc_auc = \
        compute_ROC_multiclass(label_binarize(correct_word_idx,
                                              classes=np.arange(test_confidences.shape[1])),
                               test_confidences)
    if showPlot or savePlot:
        print("3/{0:01d} Computing val ROC operating point...".format(4 if (savePlot or showPlot) else 2))
        val_OP_fpr, val_OP_tpr = get_multiclass_ROC_operating_point(correct_word_idx,
            val_confidences)
        print("4/{0:01d} Computing test ROC operating point...".format(4 if (savePlot or showPlot) else 2))
        test_OP_fpr, test_OP_tpr = get_multiclass_ROC_operating_point(correct_word_idx,
            test_confidences)
        plt.plot(val_fpr['micro'], val_tpr['micro'], color='C0', linestyle=':', linewidth=3, label='val_micro; AUC={0:0.4f}'.format(val_roc_auc['micro']))
        plt.plot(val_fpr['macro'], val_tpr['macro'], color='C0', linestyle='--', label='val_macro; AUC={0:0.4f}'.format(val_roc_auc['macro']))
        plt.plot(val_OP_fpr, val_OP_tpr, color='C0', marker='x', markersize=10)
        plt.plot(test_fpr['micro'], test_tpr['micro'], color='C1', linestyle=':', linewidth=3, label='test_micro; AUC={0:0.4f}'.format(test_roc_auc['micro']))
        plt.plot(test_fpr['macro'], test_tpr['macro'], color='C1', linestyle='--', label='test_macro; AUC={0:0.4f}'.format(test_roc_auc['macro']))
        plt.plot(test_OP_fpr, test_OP_tpr, color='C1', marker='x', markersize=10)
        plt.legend(loc='lower right')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(plot_title)
    if savePlot:
        plt.savefig('a.png')
    if showPlot:
        plt.show()
    if showPlot or savePlot:
        plt.close()
    return val_fpr, val_tpr, val_roc_auc, val_OP_fpr, val_OP_tpr, test_fpr, test_tpr, test_roc_auc, test_OP_fpr, test_OP_tpr


def get_multiclass_ROC_operating_point(correct_word_idx, confidences):
    tn, fp, fn, tp = confusion_matrix(
        label_binarize(correct_word_idx, classes=np.arange(confidences.shape[1])).ravel(),
        confidences.ravel() == np.repeat(np.max(confidences, axis=1), confidences.shape[1])
        ).ravel()
    OP_fpr = fp/(fp + tn)
    OP_tpr = tp/(tp + fn)
    return OP_fpr, OP_tpr


def compute_ROC_multiclass(y_test, y_score):
    # y_test == nxv one-hot
    # y_score == nxv full softmax scores
    n_classes = y_test.shape[1]
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # MICRO
    fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_score.ravel())
    fpr['micro'] = np.append(0, fpr['micro'])
    tpr['micro'] = np.append(0, tpr['micro'])
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # MACRO
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it
    mean_tpr /= n_classes
    all_fpr = np.append(0, all_fpr)
    mean_tpr = np.append(0, mean_tpr)
    # compute AUC
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
    return fpr, tpr, roc_auc


def compute_grid_multiclass_PR_plot_curve(correct_word_idx, preds, preds_word_idx, plotCurve=False):
    # correct_word_idx === n, {0...50}
    # preds === nxv [0, 1]
    # preds_word_idx === n, {0...50}
    # Y_test === nxv {0, 1}
    # y_score === nxv [0, 1]
    n_classes = correct_word_idx.max() + 1
    Y_test = label_binarize(correct_word_idx, classes=range(n_classes))
    y_score = preds
    y_pred = label_binarize(preds_word_idx, classes=range(n_classes))
    # TN, TP, FN, FP
    tn = {}
    fp = {}
    fn = {}
    tp = {}
    # Micro
    tn["micro"], fp["micro"], fn["micro"], tp["micro"] = \
        confusion_matrix(Y_test.ravel(), y_pred.ravel()).ravel()
    # OP
    recall_OP = {}
    recall_OP["micro"] = tp["micro"] / (tp["micro"] + fn["micro"])
    precision_OP = {}
    precision_OP["micro"] = tp["micro"] / (tp["micro"] + fp["micro"])
    print('Recall_OP["micro"]: {0:0.2f}, Precision_OP["micro"]: {1:0.2f}'.format(recall_OP["micro"], precision_OP["micro"]))
    # Macro
    recall_OP["macro"] = 0
    precision_OP["macro"] = 0
    for i in range(n_classes):
        tn[i], fp[i], fn[i], tp[i] = confusion_matrix(Y_test[:, i], y_pred[:, i]).ravel()
        recall_OP[i] = tp[i] / (tp[i] + fn[i])
        recall_OP["macro"] += recall_OP[i]
        precision_OP[i] = tp[i] / (tp[i] + fp[i])
        precision_OP["macro"] += precision_OP[i]
    # Print
    recall_OP["macro"] /= n_classes
    precision_OP["macro"] /= n_classes
    print('Recall_OP["macro"]: {0:0.2f}, Precision_OP["macro"]: {1:0.2f}'.format(recall_OP["macro"], precision_OP["macro"]))
    # sklearn
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = Y_test.shape[1]
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    precision_OP = interp(recall_OP["micro"], recall["micro"], precision["micro"])
    print('precision from sklearn at recall_OP["micro"]: {0:0.2f}'.format(precision_OP))
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))
    # Average precision score, micro-averaged over all classes: 0.98
    if plotCurve:
        # PLOT
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
        plt.show()
        plt.close()
    # Return
    return recall_OP, precision_OP, precision, recall, average_precision
