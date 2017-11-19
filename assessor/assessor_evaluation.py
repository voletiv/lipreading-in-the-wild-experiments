import glob
import os

from sklearn.metrics import confusion_matrix, roc_curve, auc, average_precision_score, precision_recall_curve

from assessor_evaluation_functions import *
from assessor_functions import *
from assessor_model import *


######################################################
# DEFINE SAVE_DIR and MODEL
######################################################


def define_save_dir_and_model(experiment_number):

    global this_assessor_save_dir, this_assessor_model

    for save_dir in sorted(glob.glob(os.path.join(ASSESSOR_SAVE_DIR, "*/"))):
        if int(save_dir.split('/')[-2][0]) == experiment_number:
            this_assessor_save_dir = save_dir
            this_assessor_model = save_dir.split('/')[-2]
            break


######################################################
# LOAD MODEL
######################################################


def load_latest_assessor():

    global this_assessor_save_dir, this_assessor_model

    assessor = read_my_model(model_file_name=os.path.join(this_assessor_save_dir, this_assessor_model+".json"),
                             weights_file_name= sorted(glob.glob(os.path.join(this_assessor_save_dir, "*.hdf5")),
                                                       key=os.path.getmtime)[-1])

    return assessor


######################################################
# PREDICT
######################################################


def predict_using_assessor(assessor, data_dir=LRW_DATA_DIR, batch_size=100, collect_type="val"):

    global this_assessor_save_dir, this_assessor_model

    # LRW_VAL
    lrw_data_generator = generate_assessor_data_batches(data_dir=data_dir, batch_size=batch_size, collect_type=collect_type,
                                                        shuffle=False, random_crop=False, verbose=False)

    # # FAST?
    # get_output = K.function([assessor.input[0], assessor.input[1], assessor.input[2], assessor.input[3], assessor.input[4], K.learning_phase()],
    #                         [assessor.output])
    # lrw_val_assessor_preds = np.array([])
    # for i in tqdm.tqdm(range(25000//batch_size)):
    #     [X, y] = next(train_generator)
    #     lrw_val_assessor_preds = np.append(lrw_val_assessor_preds, get_output([X[0], X[1], np.reshape(X[2], (batch_size, 1)), X[3], X[4], 0])[0])

    # SLOW?
    lrw_assessor_preds = np.array([])
    for i in tqdm.tqdm(range(25000//batch_size)):
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


######################################################
# EVALUATE ASSESSOR
######################################################


def evaluate_assessor(experiment_number, load_predictions=True, assessor_threshold=0.5):

    # Define assessor_save_dir and assesor_model
    define_save_dir_and_model(experiment_number)

    # Load assessor
    assessor = load_latest_assessor()

    if load_predictions:
        print("Loading predictions...")
        # Load preds
        lrw_val_assessor_preds = np.load(os.path.join(this_assessor_save_dir, this_assessor_model+"_lrw_val_preds.npy"))
        lrw_test_assessor_preds = np.load(os.path.join(this_assessor_save_dir, this_assessor_model+"_lrw_test_preds.npy"))
    else:
        print("Predicting assessor results...")
        # Predict
        lrw_val_assessor_preds = predict_using_assessor(assessor, data_dir=LRW_DATA_DIR, batch_size=100, collect_type="val")
        lrw_test_assessor_preds = predict_using_assessor(assessor, data_dir=LRW_DATA_DIR, batch_size=100, collect_type="test")

    # SOFTMAX, CORRECT_ONE_HOT_Y_ARG
    print("Loading softmax, correct_one_hot_y_arg...")
    _, lipreader_lrw_val_softmax, lrw_correct_one_hot_y_arg = load_dense_softmax_y(collect_type="val")
    _, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg = load_dense_softmax_y(collect_type="test")

    # CORRECT_OR_WRONG
    lipreader_lrw_val_correct_or_wrong = np.argmax(lipreader_lrw_val_softmax, axis=1) == lrw_correct_one_hot_y_arg
    lipreader_lrw_test_correct_or_wrong = np.argmax(lipreader_lrw_test_softmax, axis=1) == lrw_correct_one_hot_y_arg

    # Evaluate and plot ROC with Operating Point
    print("Evaluating and plot ROC with Operating Point...")
    evaluate_and_plot_ROC_with_OP(lipreader_lrw_val_correct_or_wrong, lrw_val_assessor_preds, collect_type="val",
                                  assessor_threshold=assessor_threshold, save_and_close=False)
    evaluate_and_plot_ROC_with_OP(lipreader_lrw_test_correct_or_wrong, lrw_test_assessor_preds, collect_type="test",
                                  assessor_threshold=assessor_threshold, save_and_close=True)

    # Evaluate average precision, plot PR curve
    print("Evaluatinging average precision, plot PR curve...")
    evaluate_avg_precision_plot_PR_curve(lipreader_lrw_val_correct_or_wrong, lrw_val_assessor_preds, collect_type="val", save_and_close=False)
    evaluate_avg_precision_plot_PR_curve(lipreader_lrw_test_correct_or_wrong, lrw_test_assessor_preds, collect_type="test", save_and_close=True)

    # Compare precision, recall of lipreader results and assessor-filtered results
    print("Comparing precision, recall of lipreader results and assessor-filtered results...")
    compare_PR_of_lipreader_and_assessor(lipreader_lrw_val_softmax, lrw_correct_one_hot_y_arg, lrw_val_assessor_preds, collect_type="val", assessor_threshold=assessor_threshold)
    compare_PR_of_lipreader_and_assessor(lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg, lrw_test_assessor_preds, collect_type="test", assessor_threshold=assessor_threshold)

    print("Done.")


# MAIN
if __name__ == "__main__":

    # python3 -i assessor_evaluation 4 -load

    if len(sys.argv) < 2:
        print("[ERROR] Please mention experiment_number.")
        sys.exit()

    # READ EXPERIMENT NUMBER
    experiment_number = int(sys.argv[1])
    print("Experiment number:", experiment_number)

    load_predictions = False
    if len(sys.argv) > 2:
        if sys.argv[2] == '--load-predictions' or sys.argv[2] == '-load':
            load_predictions = True

    # RUN
    evaluate_assessor(experiment_number, load_predictions=load_predictions)
