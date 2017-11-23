import numpy as np
import sys

from assessor_evaluation_functions import *


######################################################
# EVALUATE ASSESSOR
######################################################


def evaluate_assessor(experiment_number, assessor=None, load_best_or_latest_or_none='latest', assessor_threshold=0.5):

    # Define assessor_save_dir and assesor_model
    define_save_dir_and_model(experiment_number)

    # Predictions
    lrw_val_assessor_preds = load_predictions_or_predict_using_assessor(collect_type="val", batch_size=100, assessor=assessor, load_best_or_latest_or_none=load_best_or_latest_or_none)
    lrw_test_assessor_preds = load_predictions_or_predict_using_assessor(collect_type="test", batch_size=100, assessor=assessor, load_best_or_latest_or_none=load_best_or_latest_or_none)

    # SOFTMAX, CORRECT_ONE_HOT_Y_ARG
    print("Loading softmax, correct_one_hot_y_arg...")
    try:
        _, lipreader_lrw_val_softmax, lrw_correct_one_hot_y_arg = load_dense_softmax_y(collect_type="val")
        _, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg = load_dense_softmax_y(collect_type="test")
    except OSError as err:
        print(err)
        return

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

    # python3 -i assessor_evaluation 4

    if len(sys.argv) < 2:
        print("[ERROR] Please mention experiment_number.")
        sys.exit()

    # READ EXPERIMENT NUMBER
    experiment_number = int(sys.argv[1])
    print("Experiment number:", experiment_number)

    # RUN
    evaluate_assessor(experiment_number, assessor=None)
