import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def my_precision_recall(lrw_lipreader_preds_softmax, lrw_correct_one_hot_y_arg, critic_removes=None):
    lrw_lipreader_preds_correct_or_wrong = np.zeros((25000, 500))
    # Correct or wrong
    for w in range(500):
        lrw_lipreader_preds_correct_or_wrong[w*50:(w+1)*50, lrw_correct_one_hot_y_arg[w*50]] = 1
    # P-R
    lrw_lipreader_precision_w = np.zeros((25000, 500))
    lrw_lipreader_recall_w = np.zeros((25000, 500))
    lrw_lipreader_avg_precision_w = np.zeros((500))
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
        lrw_lipreader_precision_w[:, w] = np.cumsum(lrw_lipreader_preds_correct_or_wrong_sorted_w)/(np.arange(500*50)+1)
        lrw_lipreader_recall_w[:, w] = np.cumsum(lrw_lipreader_preds_correct_or_wrong_sorted_w)/50
        lrw_lipreader_avg_precision_w[w] = np.sum(np.cumsum(lrw_lipreader_preds_correct_or_wrong_sorted_w) / (np.arange(500*50)+1) * lrw_lipreader_preds_correct_or_wrong_sorted_w)/50
    # Array
    return lrw_lipreader_precision_w, lrw_lipreader_recall_w, lrw_lipreader_avg_precision_w


def plot_ROC_with_OP(fpr, tpr, roc_auc, fpr_op, tpr_op, assessor_save_dir, this_model, lrw_type, threshold=.5, save_and_close=False):
    plt.plot(fpr, tpr, lw=1, label='LRW ' + lrw_type +' ROC (AUC = %0.2f)' % (roc_auc))
    plt.plot(fpr_op, tpr_op, marker='x', markersize=10, color='C4')
    plt.title("ROC curve " + this_model + "_thresh" + str(threshold))
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    # plt.show()
    if save_and_close:
        print("Saving ROC:", os.path.join(assessor_save_dir, this_model+"_ROC_thresh"+str(threshold)+".png"))
        plt.savefig(os.path.join(assessor_save_dir, this_model+"_ROC_thresh"+str(threshold)+".png"))
        plt.close()


def plot_assessor_PR_curve(recall, precision, average_precision, assessor_save_dir, this_model, lrw_type="val", save_and_close=False):
    plt.step(recall, precision, where='post', label='LRW '+lrw_type+' AP={0:0.2f}'.format(average_precision))
    plt.fill_between(recall, precision, step='post', alpha=0.2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    leg = plt.legend()
    leg.get_frame().set_alpha(0.3)
    plt.title('Precision-Recall curve')
    if save_and_close:
        print("Saving PR:", os.path.join(assessor_save_dir, this_model+"_PR.png"))
        plt.savefig(os.path.join(assessor_save_dir, this_model+"_PR.png"))
        plt.close()


def plot_P_atK_and_R_atK_vs_K(lipreader_precision_at_k_averaged_across_words, filtered_precision_at_k_averaged_across_words,
                              lipreader_recall_at_k_averaged_across_words, filtered_recall_at_k_averaged_across_words,
                              assessor_save_dir=".", this_model="assessor_cnn_adam", lrw_type="val", threshold=.5):
    # P@K vs K, R@K vs K
    plt.plot(np.arange(50)+1, lipreader_precision_at_k_averaged_across_words, label='Precision @K')
    plt.plot(np.arange(50)+1, filtered_precision_at_k_averaged_across_words, label='Assessor-filtered Precision @K')
    plt.plot(np.arange(50)+1, lipreader_recall_at_k_averaged_across_words, label='Recall @K')
    plt.plot(np.arange(50)+1, filtered_recall_at_k_averaged_across_words, label='Assessor-filteredd Recall @K')
    plt.ylim([0, 1])
    plt.legend()
    plt.xlabel("K = # of documents")
    plt.title("Precision, Recall of lipreader on LRW_"+lrw_type+", vs K")
    print("Saving P@K, R@K vs K:", os.path.join(assessor_save_dir, this_model+"_P@K_R@K_vs_K_LRW_"+lrw_type+"_thresh"+str(threshold)+".png"))
    plt.savefig(os.path.join(assessor_save_dir, this_model+"_P@K_R@K_vs_K_LRW_"+lrw_type+"_thresh"+str(threshold)+".png"))
    # plt.show()
    plt.close()


def plot_P_atK_vs_R_atK(lipreader_precision_at_k_averaged_across_words, filtered_precision_at_k_averaged_across_words,
                        lipreader_recall_at_k_averaged_across_words, filtered_recall_at_k_averaged_across_words,
                        assessor_save_dir=".", this_model="assessor_cnn_adam", lrw_type="val", threshold=.5):
    plt.plot(lipreader_recall_at_k_averaged_across_words, lipreader_precision_at_k_averaged_across_words, label="lipreader")
    plt.fill_between(lipreader_recall_at_k_averaged_across_words, lipreader_precision_at_k_averaged_across_words, step='post', alpha=0.2)
    plt.plot(filtered_recall_at_k_averaged_across_words, filtered_precision_at_k_averaged_across_words, label="Assessor-filtered lipreader")
    plt.fill_between(filtered_recall_at_k_averaged_across_words, filtered_precision_at_k_averaged_across_words, step='post', alpha=0.2)
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Recall at K")
    plt.ylabel("Precision at K")
    plt.title("P@K vs R@K curve of lipreader on LRW "+lrw_type+", till K=50")
    print("Saving P@K vs R@K:", os.path.join(assessor_save_dir, this_model+"_P@K_R@K_curve_LRW_"+lrw_type+"_thresh"+str(threshold)+".png"))
    plt.savefig(os.path.join(assessor_save_dir, this_model+"_P@K_vs_R@K_LRW_"+lrw_type+"_thresh"+str(threshold)+".png"))
    plt.close()


def plot_lrw_property_image(lrw_property, title="?????????????", cmap='jet', clim=None, save=True, assessor_save_dir=".", this_model="assessor_cnn_adam", lrw_type="val", file_name="avg_precision"):
    # lrw_property must be of shape (500,)
    # Fig
    fig, ax = plt.subplots(figsize=(28, 10))
    # Grid
    x_lim = 20
    y_lim = 25
    x, y = np.meshgrid(np.arange(x_lim), np.arange(y_lim))
    # Image
    image = plt.imshow(np.reshape(lrw_property[:x_lim*y_lim], (y_lim, x_lim)), cmap=cmap, clim=clim, aspect='auto')
    plt.colorbar(image)
    # Words in image
    for i, (x_val, y_val) in enumerate(zip(x.flatten(), y.flatten())):
        # print(image.get_array()[y_val][x_val])
        # if np.mean(image.cmap(image.get_array()[y_val][x_val])[:3]) < .5:
        #     c = 'w'
        # else:
        #     c = 'k'
        # ax.text(x_val, y_val, LRW_VOCAB[i], va='center', ha='center', fontsize=7, color=c)
        ax.text(x_val, y_val, LRW_VOCAB[i], va='center', ha='center', fontsize=7)
    # ax.set_xlim(0, x_lim)
    # ax.set_ylim(0, y_lim)
    # ax.set_xticks(np.arange(x_lim))
    # ax.set_yticks(np.arange(y_lim))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.grid()
    plt.title(title)
    if save:
        plt.savefig(os.path.join(assessor_save_dir, this_model+"_"+file_name+"_"+lrw_type+".png"))
    else:
        plt.show()
    plt.close()
