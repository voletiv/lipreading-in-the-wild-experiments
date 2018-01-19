import numpy as np

from sklearn.model_selection import train_test_split

from assessor_nonDeep_functions import *

#####################################
# DATA
#####################################

X_train, y_train, X_test, y_test, lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test = get_lrw_data()

# Orig
LRW_val_X = np.array(X_train)
LRW_test_X = np.array(X_test)

# X_train, X_test = LRW_val_X, LRW_test_X

# Full
X = np.vstack((X_train, X_test))
y = np.append(y_train, y_test)


#####################################
# SVM
#####################################

X_train, y_train, X_test, y_test, lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test = get_lrw_data()

name = "linearSVM_correctSyncnet_woLRWtrain"
assessor = 'linearSVM'
assessor_threshold = 0.5

class_weight = 'balanced'

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, X_train, X_test,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax,
                           lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           class_weight=class_weight)


name = "rbfSVM_defaultData"
assessor = 'rbfSVM'
assessor_threshold = 0.5

class_weight = 'balanced'

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, X_train, X_test,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax,
                           lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           class_weight=class_weight)

#####################################
# ENSEMBLE
#####################################

# multiple_RFs_dir = os.path.join(ASSESSOR_SAVE_DIR, 'RF_varNEstimators')
multiple_RFs_dir = os.path.join(ASSESSOR_SAVE_DIR, 'RF_varNEstimators_defTrainValTest')

X_train, y_train, X_test, y_test, lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test = get_lrw_data(use_LRW_train=True)
LRW_val_X, LRW_test_X = X_train, X_test

# Load
assessors = []
for assessor_dir in sorted(glob.glob(os.path.join(multiple_RFs_dir, "*/"))):
    assessors.append(joblib.load(glob.glob(os.path.join(assessor_dir, "*.pkl"))[0]))

lrw_val_assessor_preds = np.empty((len(LRW_val_X)))
lrw_test_assessor_preds = np.empty((len(LRW_test_X)))

# Predict
for assessor in tqdm.tqdm(assessors):
    # Predict
    pred_LRW_val = assessor.predict_proba(LRW_val_X)[:, 1]
    pred_LRW_test = assessor.predict_proba(LRW_test_X)[:, 1]
    lrw_val_assessor_preds = np.vstack((lrw_val_assessor_preds, pred_LRW_val))
    lrw_test_assessor_preds = np.vstack((lrw_test_assessor_preds, pred_LRW_test))

ensemble_assessor_save_dir = os.path.join(multiple_RFs_dir, 'ENSEMBLE')
if not os.path.exists(ensemble_assessor_save_dir):
    print("Making dir", ensemble_assessor_save_dir)
    os.makedirs(ensemble_assessor_save_dir)

np.savez(os.path.join(ensemble_assessor_save_dir, "multiple_assessors_preds"),
         lrw_val_assessor_preds=lrw_val_assessor_preds,
         lrw_test_assessor_preds=lrw_test_assessor_preds)

multiple_assessor_preds = np.load(os.path.join(ensemble_assessor_save_dir, "multiple_assessors_preds.npz"))
lrw_val_assessor_preds = multiple_assessor_preds['lrw_val_assessor_preds']
lrw_test_assessor_preds = multiple_assessor_preds['lrw_test_assessor_preds']

print("Evaluating ensemble classifier...")

ensemble_lrw_val_assessor_preds = np.mean(lrw_val_assessor_preds, axis=0)
ensemble_lrw_test_assessor_preds = np.mean(lrw_test_assessor_preds, axis=0)

evaluate_assessor(lrw_val_assessor_preds=ensemble_lrw_val_assessor_preds[25000:],
                  lrw_test_assessor_preds=ensemble_lrw_test_assessor_preds,
                  assessor="blah",
                  assessor_save_dir=ensemble_assessor_save_dir,
                  assessor_threshold=0.9,
                  lipreader_lrw_val_softmax=lipreader_lrw_val_softmax[25000:],
                  lipreader_lrw_test_softmax=lipreader_lrw_test_softmax,
                  lrw_correct_one_hot_y_arg_val=lrw_correct_one_hot_y_arg_val[25000:],
                  lrw_correct_one_hot_y_arg_test=lrw_correct_one_hot_y_arg_test)


#####################################
# MULTIPLE RFs
#####################################

X_train, y_train, X_test, y_test, lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test = get_lrw_data(use_LRW_train=True)

# 100
pre_name = "RF_withSyncnetPreds_defaultTrainValTest_estimators"
assessor = 'RF'
assessor_threshold = 0.5

test_size = 0.5
random_state = 0

for n in range(10, 100, 10):
    name = pre_name + str(n)
    n_estimators = n
    make_and_evaluate_assessor(X_train, y_train, X_test, y_test, X_train, X_test,
                               lipreader_lrw_val_softmax, lipreader_lrw_test_softmax,
                               lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test,
                               name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                               n_estimators=n_estimators, random_state=random_state)


#####################################
# RF without Dense
#####################################

# 100
name = "RF_woDense_wSyncnetPreds_defaultData_estimators50"
assessor = 'RF'
assessor_threshold = 0.5

n_estimators = 50
random_state = 0

X_train, y_train, X_test, y_test, lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test = get_lrw_data(use_dense=False)

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, X_train, X_test,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax,
                           lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           n_estimators=n_estimators, random_state=random_state)

#####################################
# RF without Dense without softmax
#####################################

# 100
name = "RF_woDense_woSM_wSyncnetPreds_defaultData_estimators50"
assessor = 'RF'
assessor_threshold = 0.5

n_estimators = 50
random_state = 0

X_train, y_train, X_test, y_test, lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test = get_lrw_data(use_dense=False, use_softmax=False)

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, X_train, X_test,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax,
                           lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           n_estimators=n_estimators, random_state=random_state)

# 10
name = "RF_woDense_woSM_wSyncnetPreds_defaultData_estimators10"
assessor = 'RF'
assessor_threshold = 0.5

n_estimators = 10
random_state = 0

X_train, y_train, X_test, y_test, lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test = get_lrw_data(use_dense=False, use_softmax=False)

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, X_train, X_test,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax,
                           lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           n_estimators=n_estimators, random_state=random_state)


#####################################
# RF with Syncnet_preds
#####################################

# 100
name = "RF_withCorrectSyncnet_withoutLRWtrain_estimators100"
assessor = 'RF'
assessor_threshold = 0.5

# test_size = 0.5
n_estimators = 100
random_state = 0

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# X_train, y_train, X_test, y_test, lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test = get_lrw_data(use_LRW_train=True, samples_per_word=200)
X_train, y_train, X_test, y_test, lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test = get_lrw_data()
LRW_val_X, LRW_test_X = X_train, X_test

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, LRW_val_X, LRW_test_X,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax,
                           lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           n_estimators=n_estimators, random_state=random_state)


# 100
name = "RF_withCorrectSyncnet_withoutLRWtrain_usingSyncnetPredsAndSoftmax_estimators100"
assessor = 'RF'
assessor_threshold = 0.5

# test_size = 0.5
n_estimators = 100
random_state = 0

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# X_train, y_train, X_test, y_test, lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test = get_lrw_data(use_LRW_train=True, samples_per_word=200)
X_train, y_train, X_test, y_test, lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test = get_lrw_data(use_dense=False, use_softmax=True, use_softmax_ratios=True)
LRW_val_X, LRW_test_X = X_train, X_test

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, LRW_val_X, LRW_test_X,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax,
                           lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           n_estimators=n_estimators, random_state=random_state)


# 100
name = "RF_withSyncnetPreds_defaultTrainTest_estimators100"
assessor = 'RF'
assessor_threshold = 0.5

test_size = 0.5
n_estimators = 100
random_state = 0

X_train, X_test = LRW_val_X, LRW_test_X

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, LRW_val_X, LRW_test_X,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           n_estimators=n_estimators, random_state=random_state)

#####################################
# RF mixData
#####################################


# 100
name = "RF_withSyncnetPreds_mix_trainTestRespective_estimators100"
assessor = 'RF'
assessor_threshold = 0.5

test_size = 0.5
n_estimators = 100
random_state = 0

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# X_train, y_train, X_test, y_test, lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg = get_lrw_data()
# LRW_val_X, LRW_test_X = X_train, X_test

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, X_train, X_test,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax,
                           lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           n_estimators=n_estimators, random_state=random_state)


# 100
name = "RF_withSyncnetPreds_dataReverse_estimators100"
assessor = 'RF'
assessor_threshold = 0.5

test_size = 0.5
n_estimators = 100
random_state = 0

# X_train, y_train, X_test, y_test, lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg = get_lrw_data()
# LRW_val_X, LRW_test_X = X_train, X_test

make_and_evaluate_assessor(X_test, y_test, X_train, y_train, LRW_test_X, LRW_val_X,
                           lipreader_lrw_test_softmax, lipreader_lrw_val_softmax, lrw_correct_one_hot_y_arg,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           n_estimators=n_estimators, random_state=random_state)

# 100
name = "RF_withSyncnetPreds_mix2525_estimators100"
assessor = 'RF'
assessor_threshold = 0.5

test_size = 0.5
n_estimators = 100
random_state = 0
mix = '2525'

X_train, y_train, X_test, y_test, lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg = get_lrw_data(mix=mix)
LRW_val_X, LRW_test_X = X_train, X_test

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, LRW_val_X, LRW_test_X,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           n_estimators=n_estimators, random_state=random_state)

# 100
name = "RF_withSyncnetPreds_mixAlternate_estimators100"
assessor = 'RF'
assessor_threshold = 0.5

test_size = 0.5
n_estimators = 100
random_state = 0
mix = 'alternate'

X_train, y_train, X_test, y_test, lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg = get_lrw_data(mix=mix)
LRW_val_X, LRW_test_X = X_train, X_test

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, LRW_val_X, LRW_test_X,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           n_estimators=n_estimators, random_state=random_state)

# 10
name = "RF_withSyncnetPreds_mix2525_estimators10"
assessor = 'RF'
assessor_threshold = 0.5

test_size = 0.5
n_estimators = 10
random_state = 0
mix = '2525'

# X_train, y_train, X_test, y_test, lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg = get_lrw_data(mix=mix)
# LRW_val_X, LRW_test_X = X_train, X_test

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, LRW_val_X, LRW_test_X,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           n_estimators=n_estimators, random_state=random_state)

# 100
name = "RF_withSyncnetPreds_testFirst0.1_estimators100"
assessor = 'RF'
assessor_threshold = 0.5

n_estimators = 100
random_state = 0
mix = 'testFirst0.1'

X_train, y_train, X_test, y_test, lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg = get_lrw_data(mix=mix)
LRW_val_X, LRW_test_X = X_train, X_test

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, LRW_val_X, LRW_test_X,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           n_estimators=n_estimators, random_state=random_state)





# n_estimator = 10
# >>> rf.score(X_train, y_train)
# 0.99151111111111112
# >>> rf.score(X_test, y_test)
# 0.74060000000000004

# n_estimator = 5
# ... rf.score(X_train, y_train)
# 0.97028888888888887
# >>> rf.score(X_test, y_test)
# 0.72299999999999998


#####################################
# RANDOM FOREST
#####################################

# 100
name = "RF100_full_testSplit0.1"
assessor = 'RF'
assessor_threshold = 0.5

test_size = 0.1
n_estimators = 100
random_state = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, LRW_val_X, LRW_test_X,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           n_estimators=n_estimators, random_state=random_state)

# ... rf.score(X_train, y_train)
# 1.0
# >>> rf.score(X_test, y_test)
# 0.76659999999999995


# 100
name = "RF_full_testSplit0.5_estimators100"
assessor = 'RF'
assessor_threshold = 0.5

test_size = 0.5
n_estimators = 100
random_state = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, LRW_val_X, LRW_test_X,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           n_estimators=n_estimators, random_state=random_state)


#####################################
# XGBOOST
#####################################

assessor = 'XGBOOST'
assessor_threshold = 0.5

# 3, 100
name = "XGBOOST_full_testSplit0.1_maxDepth3_estimators100"
max_depth = 3
n_estimators = 100
random_state = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, LRW_val_X, LRW_test_X,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           max_depth=max_depth, n_estimators=n_estimators, seed=random_state)

# 10, 10
name = "XGBOOST_full_testSplit0.1_maxDepth10_estimators10"
max_depth = 10
n_estimators = 10
random_state = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, LRW_val_X, LRW_test_X,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           max_depth=max_depth, n_estimators=n_estimators, seed=random_state)


#####################################
# GBT
#####################################

name = "GBT_full_testSplit0.1_maxDepth3_estimators20"
assessor = 'GBT'
assessor_threshold = 0.5

max_depth = 3
n_estimators = 20
random_state = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, LRW_val_X, LRW_test_X,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           max_depth=max_depth, n_estimators=n_estimators, random_state=random_state)


name = "GBT_full_testSplit0.1_maxDepth10_estimators20"
assessor = 'GBT'
assessor_threshold = 0.5

max_depth = 10
n_estimators = 20
random_state = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, LRW_val_X, LRW_test_X,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           max_depth=max_depth, n_estimators=n_estimators, random_state=random_state)


#####################################
# LINEAR SVM
#####################################

name = "linSVM_full_testSplit0.1"
assessor = 'linearSVM'
assessor_threshold = 0.5
class_weight='balanced'

random_state = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, LRW_val_X, LRW_test_X,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           n_estimators=n_estimators, class_weight=class_weight, random_state=random_state)


#####################################
# LDA
#####################################

name = "LDA_default"
assessor = 'LDA'
assessor_threshold = 0.5

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, X_train, X_test,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax,
                           lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold)

# priors (.3, .7)
name = "LDA_default_priors"
assessor = 'LDA'
assessor_threshold = 0.5

priors = (.3, .7)

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, X_train, X_test,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax,
                           lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           priors=priors)

# priors (.3, .7)
name = "LDA_woDense_priors"
assessor = 'LDA'
assessor_threshold = 0.5

priors = (.3, .7)
X_train, y_train, X_test, y_test, lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test = get_lrw_data(use_dense=False)


make_and_evaluate_assessor(X_train, y_train, X_test, y_test, X_train, X_test,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax,
                           lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           priors=priors)

# priors (.3, .7)
name = "LDA_woDense_woSM_priors"
assessor = 'LDA'
assessor_threshold = 0.5

priors = (.3, .7)
X_train, y_train, X_test, y_test, lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test = get_lrw_data(use_dense=False, use_softmax=False)


make_and_evaluate_assessor(X_train, y_train, X_test, y_test, X_train, X_test,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax,
                           lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           priors=priors)


#####################################
# QDA
#####################################

name = "QDA_default"
assessor = 'QDA'
assessor_threshold = 0.5

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, X_train, X_test,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax,
                           lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold)

# priors (.3, .7)
name = "QDA_default_priors"
assessor = 'QDA'
assessor_threshold = 0.5

priors = (.3, .7)

make_and_evaluate_assessor(X_train, y_train, X_test, y_test, X_train, X_test,
                           lipreader_lrw_val_softmax, lipreader_lrw_test_softmax,
                           lrw_correct_one_hot_y_arg_val, lrw_correct_one_hot_y_arg_test,
                           name=name, assessor=assessor, assessor_threshold=assessor_threshold,
                           priors=priors)

#####################################
# GBT, etc.
#####################################


def compare_assessors():

    n_estimator = 20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # It is important to train the ensemble of trees on a different subset
    # of the training data than the linear regression model to avoid
    # overfitting, in particular if the total number of leaves is
    # similar to the number of training samples
    X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train,
                                                                y_train,
                                                                test_size=0.1)

    # Unsupervised transformation based on totally random trees
    rt = RandomTreesEmbedding(n_estimators=n_estimator, random_state=0)

    rt_lm = LogisticRegression()
    pipeline = make_pipeline(rt, rt_lm)
    pipeline.fit(X_train, y_train)
    y_pred_rt = pipeline.predict_proba(X_test)[:, 1]
    fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)

    # Supervised transformation based on random forests
    rf = RandomForestClassifier(n_estimators=n_estimator)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

    # RF + LR
    rf_enc = OneHotEncoder()
    rf_enc.fit(rf.apply(X_train))
    rf_lm = LogisticRegression()
    rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)
    y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
    fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)

    # GBT
    grd = GradientBoostingClassifier(n_estimators=n_estimator)
    grd.fit(X_train, y_train)
    y_pred_grd = grd.predict_proba(X_test)[:, 1]
    fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)
    grd.score(X_train, y_train)
    grd.score(X_test, y_test)

    # GBT + LR
    grd_enc = OneHotEncoder()
    grd_enc.fit(grd.apply(X_train)[:, :, 0])
    grd_lm = LogisticRegression()
    grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

    y_pred_grd_lm = grd_lm.predict_proba(
        grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
    fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
    plt.plot(fpr_grd, tpr_grd, label='GBT')
    plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

