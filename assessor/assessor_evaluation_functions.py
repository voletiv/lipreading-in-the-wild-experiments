import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from sklearn.metrics import confusion_matrix, roc_curve, auc, average_precision_score, precision_recall_curve

from assessor_functions import *
from assessor_params import *
from assessor_model import *
from assessor_train_params import *


######################################################
# DEFINE SAVE_DIR and MODEL
######################################################


def define_save_dir_and_model(experiment_number):

    global this_assessor_save_dir, this_assessor_model

    for save_dir in sorted(glob.glob(os.path.join(ASSESSOR_SAVE_DIR, "[0-9]*/"))):
        if int(save_dir.split('/')[-2].split('_')[0]) == experiment_number:
            this_assessor_save_dir = save_dir
            this_assessor_model = save_dir.split('/')[-2]
            break


######################################################
# LOAD MODEL
######################################################


def load_best_or_latest_assessor(load_best_or_latest_or_none='latest'):

    global this_assessor_save_dir, this_assessor_model

    print("Loading", load_best_or_latest, "assessor...")

    if load_best_or_latest_or_none == 'latest':
        weights_file_name = sorted(glob.glob(os.path.join(this_assessor_save_dir, "*.hdf5")),
                                   key=os.path.getmtime)[-1]
    elif load_best_or_latest_or_none == 'best':
        weights_file_name = sorted(glob.glob(os.path.join(this_assessor_save_dir, "*.hdf5")),
                                   key=os.path.getmtime)[-2]

    assessor = read_my_model(model_file_name=os.path.join(this_assessor_save_dir, this_assessor_model+".json"),
                             weights_file_name=weights_file_name)

    return assessor


######################################################
# PREDICT
######################################################


def load_predictions_or_predict_using_assessor(collect_type="val", batch_size=100, assessor=None, load_best_or_latest_or_none='latest'):
    lrw_preds_file = os.path.join(this_assessor_save_dir, this_assessor_model+"_lrw_" + collect_type + "_preds.npy")

    if os.path.exists(lrw_preds_file):
        print("Loading predictions from", lrw_preds_file, "...")
        lrw_assessor_preds = np.load(lrw_preds_file)
        lrw_n_of_frames_per_sample = load_array_of_var_per_sample_from_csv(csv_file_name=N_OF_FRAMES_PER_SAMPLE_CSV_FILE, collect_type=collect_type, collect_by='sample')

        if len(lrw_assessor_preds) < len(lrw_n_of_frames_per_sample):
            print("Only", len(lrw_assessor_preds), "samples present. Predicting for", (len(lrw_n_of_frames_per_sample) - len(lrw_assessor_preds))//batch_size, "batches...")
            lrw_assessor_preds = predict_using_assessor(assessor, data_dir=LRW_DATA_DIR, batch_size=batch_size, collect_type="val",
                                                            skip_batches=len(lrw_assessor_preds)//batch_size, lrw_assessor_preds=lrw_assessor_preds)

    else:
        print("Predicting assessor results...")
        # Load assessor
        if assessor is None:
            assessor = load_best_or_latest_assessor(load_best_or_latest_or_none)
        lrw_assessor_preds = predict_using_assessor(assessor, data_dir=LRW_DATA_DIR, batch_size=batch_size, collect_type=collect_type)

    return lrw_assessor_preds


def predict_using_assessor(assessor, data_dir=LRW_DATA_DIR, batch_size=100, collect_type="val", skip_batches=0, lrw_assessor_preds=np.array([])):

    global this_assessor_save_dir, this_assessor_model

    # LRW_VAL
    lrw_data_generator = generate_assessor_data_batches(batch_size=batch_size, data_dir=data_dir, collect_type=collect_type, shuffle=False,
                                                        equal_classes=equal_classes, use_CNN_LSTM=use_CNN_LSTM, use_head_pose=use_head_pose,
                                                        grayscale_images=grayscale_images, random_crop=False, random_flip=False,
                                                        verbose=False, skip_batches=skip_batches)

    # # FAST?
    # get_output = K.function([assessor.input[0], assessor.input[1], assessor.input[2], assessor.input[3], assessor.input[4], K.learning_phase()],
    #                         [assessor.output])
    # for i in tqdm.tqdm(range(25000//batch_size)):
    #     [X, y] = next(train_generator)
    #     lrw_val_assessor_preds = np.append(lrw_val_assessor_preds, get_output([X[0], X[1], np.reshape(X[2], (batch_size, 1)), X[3], X[4], 0])[0])

    # SLOW? 
    for i in tqdm.tqdm(range(skip_batches, 25000//batch_size)):
        [X, y] = next(lrw_data_generator)
        lrw_assessor_preds = np.append(lrw_assessor_preds, assessor.predict(X))
        if (i+1) % 50 == 0:
            print("Saving", collect_type, "preds as", os.path.join(this_assessor_save_dir, this_assessor_model+"_lrw_"+collect_type+"_preds"))
            np.save(os.path.join(this_assessor_save_dir, this_assessor_model+"_lrw_"+collect_type+"_preds"), lrw_assessor_preds)

    # # Save
    # np.save(os.path.join(this_assessor_save_dir, this_assessor_model+"_lrw_"+collect_type+"_preds"), lrw_assessor_preds)

    return lrw_assessor_preds


######################################################
# ROC, OPERATING POINT
######################################################


def evaluate_and_plot_ROC_with_OP(lipreader_correct_or_wrong, lrw_assessor_preds, collect_type="val", assessor_threshold=0.5, save_and_close=True):

    global this_assessor_save_dir, this_assessor_model

    # OP
    tn, fp, fn, tp = confusion_matrix(lipreader_correct_or_wrong, lrw_assessor_preds >= assessor_threshold).ravel()
    fpr_op = fp/(fp + tn)
    tpr_op = tp/(tp + fn)

    # ROC
    fpr, tpr, thresholds = roc_curve(lipreader_correct_or_wrong, lrw_assessor_preds)
    roc_auc = auc(fpr, tpr)

    # Plot
    plot_ROC_with_OP(fpr, tpr, roc_auc, fpr_op, tpr_op,
                     this_assessor_save_dir=this_assessor_save_dir, this_assessor_model=this_assessor_model,
                     lrw_type=collect_type, assessor_threshold=assessor_threshold, save_and_close=save_and_close)


######################################################
# P-R CURVE
######################################################


def evaluate_avg_precision_plot_PR_curve(lipreader_correct_or_wrong, lrw_assessor_preds, collect_type="val", save_and_close=True):

    global this_assessor_save_dir, this_assessor_model

    # PR
    average_precision = average_precision_score(lipreader_correct_or_wrong, lrw_assessor_preds)
    precision, recall, _ = precision_recall_curve(lipreader_correct_or_wrong, lrw_assessor_preds)

    plot_assessor_PR_curve(recall, precision, average_precision,
                           this_assessor_save_dir=this_assessor_save_dir, this_assessor_model=this_assessor_model,
                           lrw_type=collect_type, save_and_close=save_and_close)


######################################################
# COMPARISON OF P-R
######################################################


def compare_PR_of_lipreader_and_assessor(lipreader_lrw_softmax, lrw_correct_one_hot_y_arg, lrw_assessor_preds, collect_type="val", assessor_threshold=0.5):

    global this_assessor_save_dir, this_assessor_model

    lipreader_lrw_precision_w, lipreader_lrw_recall_w, lipreader_lrw_avg_precision_w = \
        my_precision_recall(lipreader_lrw_softmax, lrw_correct_one_hot_y_arg)

    lipreader_lrw_precision_at_k_averaged_across_words = np.mean(lipreader_lrw_precision_w, axis=1)[:50]
    lipreader_lrw_recall_at_k_averaged_across_words = np.mean(lipreader_lrw_recall_w, axis=1)[:50]

    lrw_rejection_idx = lrw_assessor_preds <= assessor_threshold
    filtered_lipreader_lrw_precision_w, filtered_lipreader_lrw_recall_w, filtered_lipreader_lrw_avg_precision_w = \
        my_precision_recall(lipreader_lrw_softmax, lrw_correct_one_hot_y_arg, critic_removes=lrw_rejection_idx)

    filtered_lrw_precision_at_k_averaged_across_words = np.mean(filtered_lipreader_lrw_precision_w, axis=1)[:50]
    filtered_lrw_recall_at_k_averaged_across_words = np.mean(filtered_lipreader_lrw_recall_w, axis=1)[:50]

    # P@K vs K, R@K vs K
    plot_P_atK_and_R_atK_vs_K(lipreader_lrw_precision_at_k_averaged_across_words, filtered_lrw_precision_at_k_averaged_across_words,
                              lipreader_lrw_recall_at_k_averaged_across_words, filtered_lrw_recall_at_k_averaged_across_words,
                              this_assessor_save_dir=this_assessor_save_dir, this_assessor_model=this_assessor_model,
                              lrw_type=collect_type, assessor_threshold=assessor_threshold)

    # P-R curve
    plot_P_atK_vs_R_atK(lipreader_lrw_precision_at_k_averaged_across_words, filtered_lrw_precision_at_k_averaged_across_words,
                        lipreader_lrw_recall_at_k_averaged_across_words, filtered_lrw_recall_at_k_averaged_across_words,
                        this_assessor_save_dir=this_assessor_save_dir, this_assessor_model=this_assessor_model,
                        lrw_type=collect_type, assessor_threshold=assessor_threshold)

    # PRECISION GIF
    plot_lrw_property_image(lipreader_lrw_avg_precision_w, title="Average Precision (@K) - LRW "+collect_type, cmap='gray', clim=[0, 1],
        save=True, this_assessor_save_dir=this_assessor_save_dir, this_assessor_model=this_assessor_model, file_name="avg_precision", lrw_type=collect_type,
        filtered_using_assessor=False)

    plot_lrw_property_image(filtered_lipreader_lrw_avg_precision_w, title="Average Precision (@K) - LRW "+collect_type+" - filtered using assessor", cmap='gray', clim=[0, 1],
        save=True, this_assessor_save_dir=this_assessor_save_dir, this_assessor_model=this_assessor_model, file_name="avg_precision", lrw_type=collect_type,
        filtered_using_assessor=True)


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


def plot_ROC_with_OP(fpr, tpr, roc_auc, fpr_op, tpr_op, this_assessor_save_dir, this_assessor_model, lrw_type, assessor_threshold=.5, save_and_close=False):
    plt.plot(fpr, tpr, lw=1, label='LRW ' + lrw_type +' ROC (AUC = %0.2f)' % (roc_auc))
    plt.plot(fpr_op, tpr_op, marker='x', markersize=10, color='C4')
    plt.title("ROC curve " + this_assessor_model + "_thresh" + str(assessor_threshold))
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    # plt.show()
    if save_and_close:
        print("Saving ROC:", os.path.join(this_assessor_save_dir, this_assessor_model+"_ROC_thresh"+str(assessor_threshold)+".png"))
        plt.savefig(os.path.join(this_assessor_save_dir, this_assessor_model+"_ROC_thresh"+str(assessor_threshold)+".png"))
        plt.close()
        time.sleep(.5)


def plot_assessor_PR_curve(recall, precision, average_precision, this_assessor_save_dir, this_assessor_model, lrw_type="val", save_and_close=False):
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
        print("Saving PR:", os.path.join(this_assessor_save_dir, this_assessor_model+"_PR.png"))
        plt.savefig(os.path.join(this_assessor_save_dir, this_assessor_model+"_PR.png"))
        plt.close()


def plot_P_atK_and_R_atK_vs_K(lipreader_precision_at_k_averaged_across_words, filtered_precision_at_k_averaged_across_words,
                              lipreader_recall_at_k_averaged_across_words, filtered_recall_at_k_averaged_across_words,
                              this_assessor_save_dir=".", this_assessor_model="assessor_cnn_adam", lrw_type="val", assessor_threshold=.5):
    # P@K vs K, R@K vs K
    plt.plot(np.arange(50)+1, lipreader_precision_at_k_averaged_across_words, label='Precision @K')
    plt.plot(np.arange(50)+1, filtered_precision_at_k_averaged_across_words, label='Assessor-filtered Precision @K')
    plt.plot(np.arange(50)+1, lipreader_recall_at_k_averaged_across_words, label='Recall @K')
    plt.plot(np.arange(50)+1, filtered_recall_at_k_averaged_across_words, label='Assessor-filteredd Recall @K')
    plt.ylim([0, 1])
    plt.legend()
    plt.xlabel("K = # of documents")
    plt.title("Precision, Recall of lipreader on LRW_"+lrw_type+", vs K")
    print("Saving P@K, R@K vs K:", os.path.join(this_assessor_save_dir, this_assessor_model+"_P@K_R@K_vs_K_LRW_"+lrw_type+"_thresh"+str(assessor_threshold)+".png"))
    plt.savefig(os.path.join(this_assessor_save_dir, this_assessor_model+"_P@K_R@K_vs_K_LRW_"+lrw_type+"_thresh"+str(assessor_threshold)+".png"))
    # plt.show()
    plt.close()


def plot_P_atK_vs_R_atK(lipreader_precision_at_k_averaged_across_words, filtered_precision_at_k_averaged_across_words,
                        lipreader_recall_at_k_averaged_across_words, filtered_recall_at_k_averaged_across_words,
                        this_assessor_save_dir=".", this_assessor_model="assessor_cnn_adam", lrw_type="val", assessor_threshold=.5):
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
    print("Saving P@K vs R@K:", os.path.join(this_assessor_save_dir, this_assessor_model+"_P@K_R@K_curve_LRW_"+lrw_type+"_thresh"+str(assessor_threshold)+".png"))
    plt.savefig(os.path.join(this_assessor_save_dir, this_assessor_model+"_P@K_vs_R@K_LRW_"+lrw_type+"_thresh"+str(assessor_threshold)+".png"))
    plt.close()


def plot_lrw_property_image(lrw_property, title="?????????????", cmap='jet', clim=None, save=True,
                            this_assessor_save_dir=".", this_assessor_model="assessor_cnn_adam",
                            file_name="avg_precision", lrw_type="val", filtered_using_assessor=False):
    # lrw_property must be of shape (500,)
    # Grid
    x_lim = 20
    y_lim = 25
    x, y = np.meshgrid(np.arange(x_lim), np.arange(y_lim))
    # Fig
    fig, ax = plt.subplots(figsize=(25, 15))
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
        ax.text(x_val, y_val, LRW_VOCAB[i], va='center', ha='center')
    # ax.set_xlim(0, x_lim)
    # ax.set_ylim(0, y_lim)
    # ax.set_xticks(np.arange(x_lim))
    # ax.set_yticks(np.arange(y_lim))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.grid()
    plt.title(title)
    if save:
        fig_title = os.path.join(this_assessor_save_dir, this_assessor_model+"_lipreader_"+file_name+"_"+lrw_type)
        if filtered_using_assessor:
            fig_title += "_assessor_filtered.png"
        else:
            fig_title += ".png"
        print("Saving lrw_property_image:", fig_title)
        plt.savefig(fig_title)
    else:
        plt.show()
    plt.close()
