import numpy as np
# import optunity
# import optunity.metrics
import tqdm

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition

from assessor_evaluation_functions import *

# Number of frames
lrw_val_n_of_frames = load_array_of_var_per_sample_from_csv(csv_file_name=N_OF_FRAMES_PER_SAMPLE_CSV_FILE, collect_type='val', collect_by='sample')
lrw_test_n_of_frames = load_array_of_var_per_sample_from_csv(csv_file_name=N_OF_FRAMES_PER_SAMPLE_CSV_FILE, collect_type='test', collect_by='sample')

# Dense, Softmax, One_hot_y_arg
lipreader_lrw_val_dense, lipreader_lrw_val_softmax, lrw_correct_one_hot_y_arg = load_dense_softmax_y(collect_type="val")
lipreader_lrw_test_dense, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg = load_dense_softmax_y(collect_type="test")

lipreader_lrw_val_correct_or_wrong = np.argmax(lipreader_lrw_val_softmax, axis=1) == lrw_correct_one_hot_y_arg
lipreader_lrw_test_correct_or_wrong = np.argmax(lipreader_lrw_test_softmax, axis=1) == lrw_correct_one_hot_y_arg

#####################################
# LINEAR UNOPT
#####################################

# train model on the full training set with tuned hyperparameters
SVM_linear = SVC(kernel='linear', class_weight='balanced').fit(lipreader_lrw_val_dense[:1000], lipreader_lrw_val_correct_or_wrong[:1000])

# Save
joblib.dump(SVM_linear, os.path.join(ASSESSOR_SAVE_DIR, 'SVM_linear_optimal.pkl'), compress=3) 

# Acc
SVM_linear_optimal.score(lipreader_lrw_val_dense, lipreader_lrw_val_correct_or_wrong)
SVM_linear_optimal.score(lipreader_lrw_test_dense, lipreader_lrw_test_correct_or_wrong)


#####################################
# RBF UNOPT
#####################################

# train model on the full training set with tuned hyperparameters
SVM_linear_optimal = SVC(kernel='linear', class_weight='balanced').fit(lipreader_lrw_val_dense[:1000], lipreader_lrw_val_correct_or_wrong[:1000])

# Save
joblib.dump(SVM_linear_optimal, os.path.join(ASSESSOR_SAVE_DIR, 'SVM_linear_optimal.pkl'), compress=3) 

# Acc
SVM_linear_optimal.score(lipreader_lrw_val_dense, lipreader_lrw_val_correct_or_wrong)
SVM_linear_optimal.score(lipreader_lrw_test_dense, lipreader_lrw_test_correct_or_wrong)


#####################################
# LINEAR OPT
#####################################

# score function: twice iterated 10-fold cross-validated accuracy
@optunity.cross_validated(x=lipreader_lrw_val_dense, y=lipreader_lrw_val_correct_or_wrong, num_folds=2, num_iter=1)
def svm_linear_auc(x_train, y_train, x_test, y_test, logC, logGamma):
    model = SVC(kernel='linear', C=10 ** logC, gamma=10 ** logGamma, class_weight='balanced').fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)

# perform tuning on linear
hps_linear, _, _ = optunity.maximize(svm_linear_auc, num_evals=10, logC=[-5, 2], logGamma=[-5, 1])
print(hps_linear)

# train model on the full training set with tuned hyperparameters
SVM_linear_optimal = SVC(kernel='linear', C=10 ** hps_linear['logC'], gamma=10 ** hps_linear['logGamma'], class_weight='balanced', probability=True).fit(lipreader_lrw_val_dense, lipreader_lrw_val_correct_or_wrong)

# Save
joblib.dump(SVM_linear_optimal, os.path.join(ASSESSOR_SAVE_DIR, 'SVM_linear_optimal.pkl'), compress=3) 

# Acc
SVM_linear_optimal.score(lipreader_lrw_val_dense, lipreader_lrw_val_correct_or_wrong)
SVM_linear_optimal.score(lipreader_lrw_test_dense, lipreader_lrw_test_correct_or_wrong)
# >>> # Acc
# ... SVM_linear_optimal.score(train_matrix, train_lipreader_preds_correct_or_wrong)
# 0.73557257592681236
# >>> SVM_linear_optimal.score(val_matrix, val_lipreader_preds_correct_or_wrong)
# 0.78969957081545061
# >>> SVM_linear_optimal.score(si_matrix, si_lipreader_preds_correct_or_wrong)
# 0.54808333333333337

# CONFUSION MATRIX, OPERATING POINT
train_SVM_linear_opt_OP_fpr, train_SVM_linear_opt_OP_tpr, \
        val_SVM_linear_opt_OP_fpr, val_SVM_linear_opt_OP_tpr, \
        si_SVM_linear_opt_OP_fpr, si_SVM_linear_opt_OP_tpr = \
    calc_grid_operating_points(SVM_linear_optimal,
        train_lipreader_preds_correct_or_wrong, val_lipreader_preds_correct_or_wrong, si_lipreader_preds_correct_or_wrong,
        train_matrix, val_matrix, si_matrix)

# Scores
train_SVM_linear_opt_score = SVM_linear_optimal.decision_function(train_matrix)
val_SVM_linear_opt_score = SVM_linear_optimal.decision_function(val_matrix)
si_SVM_linear_opt_score = SVM_linear_optimal.decision_function(si_matrix)

# Compute ROC
train_SVM_linear_opt_fpr, train_SVM_linear_opt_tpr, train_SVM_linear_opt_thresholds, train_SVM_linear_opt_roc_auc, \
        val_SVM_linear_opt_fpr, val_SVM_linear_opt_tpr, val_SVM_linear_opt_thresholds, val_SVM_linear_opt_roc_auc, \
        si_SVM_linear_opt_fpr, si_SVM_linear_opt_tpr, si_SVM_linear_opt_thresholds, si_SVM_linear_opt_roc_auc = \
    compute_ROC_grid_singleclass(train_lipreader_preds_correct_or_wrong, train_SVM_linear_opt_score,
        val_lipreader_preds_correct_or_wrong, val_SVM_linear_opt_score,
        si_lipreader_preds_correct_or_wrong, si_SVM_linear_opt_score,
        train_SVM_linear_opt_OP_fpr, train_SVM_linear_opt_OP_tpr,
        val_SVM_linear_opt_OP_fpr, val_SVM_linear_opt_OP_tpr,
        si_SVM_linear_opt_OP_fpr, si_SVM_linear_opt_OP_tpr,
        savePlot=True, showPlot=True,
        plot_title='ROC curve of linear SVM (optimized)')

# FINDING OPTIMAL ROC OPERATING POINT

# Old fpr, tpr, acc
train_SVM_linear_opt_OP_fpr, train_SVM_linear_opt_OP_tpr
# (0.19528849436667806, 0.7297320681798518, 0.73557257592681236)
val_SVM_linear_opt_OP_fpr, val_SVM_linear_opt_OP_tpr
# (0.23053892215568864, 0.7914507772020726, 0.78969957081545061)
si_SVM_linear_opt_OP_fpr, si_SVM_linear_opt_OP_tpr
# (0.6181177108760509, 0.8130134025075659, 0.54808333333333337)

# Finding optimal point, accs
SVM_linear_opt_optimalOP_threshold, train_SVM_linear_opt_optimalOP_fpr, train_SVM_linear_opt_optimalOP_tpr, train_SVM_linear_opt_optimalOP_acc = \
    find_ROC_optimalOP(train_SVM_linear_opt_fpr, train_SVM_linear_opt_tpr, train_SVM_linear_opt_thresholds, train_SVM_linear_opt_score, train_lipreader_preds_correct_or_wrong)

val_SVM_linear_opt_optimalOP_fpr, val_SVM_linear_opt_optimalOP_tpr, val_SVM_linear_opt_optimalOP_acc = find_fpr_tpr_acc_from_thresh(val_lipreader_preds_correct_or_wrong, val_SVM_linear_opt_score, optimalOP_threshold)
si_SVM_linear_opt_optimalOP_fpr, si_SVM_linear_opt_optimalOP_tpr, si_SVM_linear_opt_optimalOP_acc = find_fpr_tpr_acc_from_thresh(si_lipreader_preds_correct_or_wrong, si_SVM_linear_opt_score, optimalOP_threshold)

# New fpr, tpr, acc
train_SVM_linear_opt_optimalOP_fpr, train_SVM_linear_opt_optimalOP_tpr, train_SVM_linear_opt_optimalOP_acc
# (0.21338340730624786, 0.74937271075476597, 0.75227381522259451)
val_SVM_linear_opt_optimalOP_fpr, val_SVM_linear_opt_optimalOP_tpr, val_SVM_linear_opt_optimalOP_acc
# (0.25449101796407186, 0.80569948186528495, 0.80090605627086309)
si_SVM_linear_opt_optimalOP_fpr, si_SVM_linear_opt_optimalOP_tpr, si_SVM_linear_opt_optimalOP_acc
# (0.64144290751288313, 0.83073929961089499, 0.5405833333333333)

plot_grid_ROC(train_SVM_linear_opt_fpr, train_SVM_linear_opt_tpr, train_SVM_linear_opt_roc_auc,
        val_SVM_linear_opt_fpr, val_SVM_linear_opt_tpr, val_SVM_linear_opt_roc_auc,
        si_SVM_linear_opt_fpr, si_SVM_linear_opt_tpr, si_SVM_linear_opt_roc_auc,
        train_OP_fpr=train_SVM_linear_opt_OP_fpr, train_OP_tpr=train_SVM_linear_opt_OP_tpr,
        val_OP_fpr=val_SVM_linear_opt_OP_fpr, val_OP_tpr=val_SVM_linear_opt_OP_tpr,
        si_OP_fpr=si_SVM_linear_opt_OP_fpr, si_OP_tpr=si_SVM_linear_opt_OP_tpr,
        train_optimalOP_fpr=train_SVM_linear_opt_optimalOP_fpr, train_optimalOP_tpr=train_SVM_linear_opt_optimalOP_tpr,
        val_optimalOP_fpr=val_SVM_linear_opt_optimalOP_fpr, val_optimalOP_tpr=val_SVM_linear_opt_optimalOP_tpr,
        si_optimalOP_fpr=si_SVM_linear_opt_optimalOP_fpr, si_optimalOP_tpr=si_SVM_linear_opt_optimalOP_tpr,
        plot_title='ROC curve of linear SVM (optimized)')

np.savez('ROC_SVM_linear_opt',
    train_SVM_linear_opt_score=train_SVM_linear_opt_score, val_SVM_linear_opt_score=val_SVM_linear_opt_score, si_SVM_linear_opt_score=si_SVM_linear_opt_score,
    train_SVM_linear_opt_fpr=train_SVM_linear_opt_fpr, train_SVM_linear_opt_tpr=train_SVM_linear_opt_tpr, train_SVM_linear_opt_thresholds=train_SVM_linear_opt_thresholds, train_SVM_linear_opt_roc_auc=train_SVM_linear_opt_roc_auc,
    val_SVM_linear_opt_fpr=val_SVM_linear_opt_fpr, val_SVM_linear_opt_tpr=val_SVM_linear_opt_tpr, val_SVM_linear_opt_thresholds=val_SVM_linear_opt_thresholds, val_SVM_linear_opt_roc_auc=val_SVM_linear_opt_roc_auc,
    si_SVM_linear_opt_fpr=si_SVM_linear_opt_fpr, si_SVM_linear_opt_tpr=si_SVM_linear_opt_tpr, si_SVM_linear_opt_thresholds=si_SVM_linear_opt_thresholds, si_SVM_linear_opt_roc_auc=si_SVM_linear_opt_roc_auc,
    train_SVM_linear_opt_OP_fpr=train_SVM_linear_opt_OP_fpr, train_SVM_linear_opt_OP_tpr=train_SVM_linear_opt_OP_tpr,
    val_SVM_linear_opt_OP_fpr=val_SVM_linear_opt_OP_fpr, val_SVM_linear_opt_OP_tpr=val_SVM_linear_opt_OP_tpr,
    si_SVM_linear_opt_OP_fpr=si_SVM_linear_opt_OP_fpr, si_SVM_linear_opt_OP_tpr=si_SVM_linear_opt_OP_tpr,
    SVM_linear_opt_optimalOP_threshold=SVM_linear_opt_optimalOP_threshold,
    train_SVM_linear_opt_optimalOP_fpr=train_SVM_linear_opt_optimalOP_fpr, train_SVM_linear_opt_optimalOP_tpr=train_SVM_linear_opt_optimalOP_tpr,
    val_SVM_linear_opt_optimalOP_fpr=val_SVM_linear_opt_optimalOP_fpr, val_SVM_linear_opt_optimalOP_tpr=val_SVM_linear_opt_optimalOP_tpr,
    si_SVM_linear_opt_optimalOP_fpr=si_SVM_linear_opt_optimalOP_fpr, si_SVM_linear_opt_optimalOP_tpr=si_SVM_linear_opt_optimalOP_tpr)

