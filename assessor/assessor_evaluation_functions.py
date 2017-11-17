

def my_precision_recall(lrw_lipreader_preds_softmax, lrw_correct_one_hot_y_arg, critic_removes=None):
    lrw_lipreader_preds_correct_or_wrong = np.zeros((25000, 500))
    # Correct or wrong
    for w in range(500):
        lrw_lipreader_preds_correct_or_wrong[w*50:(w+1)*50, lrw_correct_one_hot_y_arg[w*50]] = 1
    # P-R
    lrw_lipreader_precision_w = np.zeros((25000, 500))
    lrw_lipreader_recall_w = np.zeros((25000, 500))
    for w in range(500):
        # Sort softmax for that word
        lrw_lipreader_preds_softmax_argsort_w = np.argsort(lrw_lipreader_preds_softmax[:, lrw_correct_one_hot_y_arg[w*50]])[::-1]
        # Sort correct_or_wrong
        lrw_lipreader_preds_correct_or_wrong_sorted_w = lrw_lipreader_preds_correct_or_wrong[:, lrw_correct_one_hot_y_arg[w*50]][lrw_lipreader_preds_softmax_argsort_w]
        if critic_removes is not None:
            critic_removes_w = critic_removes[lrw_lipreader_preds_softmax_argsort_w]
            lrw_lipreader_preds_correct_or_wrong_sorted_w_selects = lrw_lipreader_preds_correct_or_wrong_sorted_w[np.logical_not(critic_removes_w)]
            lrw_lipreader_preds_correct_or_wrong_sorted_w_rejects = lrw_lipreader_preds_correct_or_wrong_sorted_w[critic_removes_w]
            lrw_lipreader_preds_correct_or_wrong_sorted_w = np.concatenate([lrw_lipreader_preds_correct_or_wrong_sorted_w_selects[:50],
                                                                        lrw_lipreader_preds_correct_or_wrong_sorted_w_rejects,
                                                                        lrw_lipreader_preds_correct_or_wrong_sorted_w_selects[50:]])
        # P-R
        lrw_lipreader_precision_w[:, lrw_correct_one_hot_y_arg[w*50]] = np.cumsum(lrw_lipreader_preds_correct_or_wrong_sorted_w)/(np.arange(500*50)+1)
        lrw_lipreader_recall_w[:, lrw_correct_one_hot_y_arg[w*50]] = np.cumsum(lrw_lipreader_preds_correct_or_wrong_sorted_w)/50
    # Array
    return lrw_lipreader_precision_w, lrw_lipreader_recall_w


def plot_ROC_with_OP(fpr, tpr, roc_auc, fpr_op, tpr_op, assessor_save_dir, this_model, threshold=.5):
    plt.plot(fpr, tpr, lw=1, label='ROC (AUC = %0.2f)' % (roc_auc))
    plt.plot(fpr_op, tpr_op, marker='x', markersize=10, color='C0')
    plt.title("ROC curve " + this_model + " " + str(threshold))
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(assessor_save_dir, this_model+"_ROC_"+str(threshold)+".png"))
    plt.close()


def plot_PR_curve(recall, precision, average_precision, assessor_save_dir, this_model):
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
    plt.savefig(os.path.join(assessor_save_dir, this_model+"_PR"))
    plt.close()


def plot_P_atK_and_R_atK_vs_K(lipreader_precision_at_k_averaged_across_words, filtered_precision_at_k_averaged_across_words,
                              lipreader_recall_at_k_averaged_across_words, filtered_recall_at_k_averaged_across_words,
                              assessor_save_dir=".", this_model="assessor_cnn_adam", lrw_type="val"):
    # P@K vs K, R@K vs K
    plt.plot(np.arange(50)+1, lipreader_precision_at_k_averaged_across_words, label='Precision @K')
    plt.plot(np.arange(50)+1, filtered_precision_at_k_averaged_across_words, label='Assessor-filtered Precision @K')
    plt.plot(np.arange(50)+1, lipreader_recall_at_k_averaged_across_words, label='Recall @K')
    plt.plot(np.arange(50)+1, filtered_recall_at_k_averaged_across_words, label='Assessor-filteredd Recall @K')
    plt.ylim([0, 1])
    plt.legend()
    plt.xlabel("K = # of documents")
    plt.title("Precision, Recall of lipreader on LRW_val, vs K")
    plt.savefig(os.path.join(assessor_save_dir, this_model+"_P@K_R@K_vs_K_LRW_"+lrw_type+".png"))
    # plt.show()
    plt.close()


def plot_P_atK_vs_R_atK(lipreader_precision_at_k_averaged_across_words, filtered_precision_at_k_averaged_across_words,
                        lipreader_recall_at_k_averaged_across_words, filtered_recall_at_k_averaged_across_words,
                        assessor_save_dir=".", this_model="assessor_cnn_adam", lrw_type="val"):
    plt.plot(lipreader_recall_at_k_averaged_across_words, lipreader_precision_at_k_averaged_across_words, label="lipreader")
    plt.plot(filtered_recall_at_k_averaged_across_words, filtered_precision_at_k_averaged_across_words, label="Assessor-filtered lipreader")
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Recall at K")
    plt.ylabel("Precision at K")
    plt.title("P@K vs R@K curve of lipreader on LRW val, till K=50")
    plt.savefig(os.path.join(assessor_save_dir, this_model+"_PR_curve_LRW_"+lrw_type+".png"))
    plt.close()
