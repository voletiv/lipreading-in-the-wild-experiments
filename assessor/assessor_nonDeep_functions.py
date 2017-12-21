from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import KFold
from sklearn.externals import joblib

from xgboost import XGBClassifier

from assessor_evaluation import *


#####################################
# Make and evaluate assessor
#####################################


def make_and_evaluate_assessor(X_train, y_train, X_test, y_test, LRW_val_X, LRW_test_X,
                               lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg,
                               name='', assessor='XGBOOST', assessor_threshold=0.5,
                               k_fold=False, n_splits=10, **kwargs):

    SAVE_DIR = os.path.join(ASSESSOR_SAVE_DIR, name)

    # Make the dir if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        print("Making dir", SAVE_DIR)
        os.makedirs(SAVE_DIR)

    # CLF
    if assessor == 'linearRressor':
        pass
    elif assessor == 'LogisticRegressor':
        pass
    elif assessor == 'linearSVM':
        SVC(kernel='linear', **kwargs)  # class_weight='balanced'
    elif assessor == 'rbfSVM':
        SVC(kernel='rbf', **kwargs)     # class_weight='balanced'
    elif assessor == 'RF':
        clf = RandomForestClassifier(**kwargs)  # n_estimators
    elif assessor == 'GBT':
        clf = GradientBoostingClassifier(**kwargs)  # n_estimators
    elif assessor == 'XGBOOST':
        clf = XGBClassifier(**kwargs)

    print("Fitting classifier...")
    # Fit
    if k_fold:
        kf = KFold(n_splits=10)
        # gkf.get_n_splits(X_train, y_train)
        for train_index, test_index in tqdm.tqdm(kf.split(X_train)):
            clf.fit(X_train[train_index], y_train[train_index])
    else:
        clf.fit(X_train, y_train)

    # Score
    print(clf.score(X_train, y_train))
    print(clf.score(X_test, y_test))

    # Save
    print("Saving classifier...")
    joblib.dump(clf, os.path.join(SAVE_DIR, name+'.pkl'), compress=3)

    # Predict
    pred_LRW_val = clf.predict_proba(LRW_val_X)[:, 1]
    pred_LRW_test = clf.predict_proba(LRW_test_X)[:, 1]

    print("Evaluating classifier...")
    # Evaluate
    evaluate_assessor(lrw_val_assessor_preds=pred_LRW_val,
                      lrw_test_assessor_preds=pred_LRW_test,
                      assessor=assessor,
                      assessor_save_dir=SAVE_DIR,
                      assessor_threshold=assessor_threshold,
                      lipreader_lrw_val_softmax=lipreader_lrw_val_softmax,
                      lipreader_lrw_test_softmax=lipreader_lrw_test_softmax,
                      lrw_correct_one_hot_y_arg=lrw_correct_one_hot_y_arg)


#####################################
# Data
#####################################


def get_lrw_data(use_syncnet_preds=True, use_pca=False, use_softmax=True, use_softmax_ratios=True, mix=None):

    # Number of frames
    lrw_val_n_of_frames = load_array_of_var_per_sample_from_csv(csv_file_name=N_OF_FRAMES_PER_SAMPLE_CSV_FILE, collect_type='val', collect_by='sample')
    lrw_test_n_of_frames = load_array_of_var_per_sample_from_csv(csv_file_name=N_OF_FRAMES_PER_SAMPLE_CSV_FILE, collect_type='test', collect_by='sample')

    if use_syncnet_preds:
        # Syncnet preds
        lrw_val_syncnet_preds_full = load_syncnet_preds(collect_type='val')
        lrw_test_syncnet_preds_full = load_syncnet_preds(collect_type='test')

        lrw_val_syncnet_preds = np.zeros((lrw_val_syncnet_preds_full.shape[0], lrw_val_syncnet_preds_full.shape[2]))
        lrw_test_syncnet_preds = np.zeros((lrw_test_syncnet_preds_full.shape[0], lrw_test_syncnet_preds_full.shape[2]))

        for i in range(len(lrw_val_syncnet_preds)):
            lrw_val_syncnet_preds[i] = lrw_val_syncnet_preds_full[i][-(lrw_val_n_of_frames[i]//2 + 1)]

        for i in range(len(lrw_test_syncnet_preds)):
            lrw_test_syncnet_preds[i] = lrw_test_syncnet_preds_full[i][-(lrw_test_n_of_frames[i]//2 + 1)]

    # Dense, Softmax, One_hot_y_arg
    lipreader_lrw_val_dense, lipreader_lrw_val_softmax, lrw_correct_one_hot_y_arg = load_dense_softmax_y(collect_type="val")
    lipreader_lrw_test_dense, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg = load_dense_softmax_y(collect_type="test")

    lipreader_lrw_val_correct_or_wrong = np.argmax(lipreader_lrw_val_softmax, axis=1) == lrw_correct_one_hot_y_arg
    lipreader_lrw_test_correct_or_wrong = np.argmax(lipreader_lrw_test_softmax, axis=1) == lrw_correct_one_hot_y_arg

    lipreader_lrw_val_softmax_ratios = load_softmax_ratios(collect_type="val")
    lipreader_lrw_test_softmax_ratios = load_softmax_ratios(collect_type="test")

    if use_pca:
        # PCA
        pca_dense = decomposition.PCA(n_components=32)
        pca_dense.fit(lipreader_lrw_val_dense)
        lipreader_lrw_val_dense = pca_dense.transform(lipreader_lrw_val_dense)
        lipreader_lrw_test_dense = pca_dense.transform(lipreader_lrw_test_dense)
        pca_softmax = decomposition.PCA(n_components=32)
        pca_softmax.fit(lipreader_lrw_val_softmax)
        lipreader_lrw_val_softmax = pca_softmax.transform(lipreader_lrw_val_softmax)
        lipreader_lrw_test_softmax = pca_softmax.transform(lipreader_lrw_test_softmax)

    X_train = np.hstack((np.reshape(lrw_val_n_of_frames, (len(lrw_val_n_of_frames), 1)), lipreader_lrw_val_dense))
    y_train = lipreader_lrw_val_correct_or_wrong
    X_test = np.hstack((np.reshape(lrw_test_n_of_frames, (len(lrw_test_n_of_frames), 1)), lipreader_lrw_test_dense))
    y_test = lipreader_lrw_test_correct_or_wrong

    if use_syncnet_preds:
        X_train = np.hstack((lrw_val_syncnet_preds, X_train))
        X_test = np.hstack((lrw_test_syncnet_preds, X_test))

    if use_softmax:
        X_train = np.hstack((X_train, lipreader_lrw_val_softmax))
        X_test = np.hstack((X_test, lipreader_lrw_test_softmax))

    if use_softmax_ratios:
        X_train = np.hstack((X_train, lipreader_lrw_val_softmax_ratios))
        X_test = np.hstack((X_test, lipreader_lrw_test_softmax_ratios))

    # # Choose 50-50
    # pos_X = train_features[np.where(lipreader_lrw_val_correct_or_wrong == 1)]
    # pos_y = lipreader_lrw_val_correct_or_wrong[np.where(lipreader_lrw_val_correct_or_wrong == 1)]
    # neg_X = train_features[np.where(lipreader_lrw_val_correct_or_wrong == 0)]
    # neg_y = lipreader_lrw_val_correct_or_wrong[np.where(lipreader_lrw_val_correct_or_wrong == 0)]
    # X_train_eq = np.vstack((pos_X[:len(neg_X)], neg_X))
    # y_train_eq = np.append(pos_y[:len(neg_X)], neg_y)
    # X_train, X_val, y_train, y_val = train_test_split(X_train_eq, y_train_eq, test_size=0.33, random_state=0)

    if mix is not None:
        X_train_old = np.array(X_train)
        y_train_old = np.array(y_train)
        X_test_old = np.array(X_test)
        y_test_old = np.array(y_test)
        lipreader_lrw_val_softmax_old = np.array(lipreader_lrw_val_softmax)
        lipreader_lrw_test_softmax_old = np.array(lipreader_lrw_test_softmax)

        if mix == '2525':
            print('25-25 mixing!')
            for i in range(500):
                X_train[i*50+25:(i+1)*50] = X_test_old[i*50+25:(i+1)*50]
                X_test[i*50+25:(i+1)*50] = X_train_old[i*50+25:(i+1)*50]
                lipreader_lrw_val_softmax[i*50+25:(i+1)*50] = lipreader_lrw_test_softmax_old[i*50+25:(i+1)*50]
                lipreader_lrw_test_softmax[i*50+25:(i+1)*50] = lipreader_lrw_val_softmax_old[i*50+25:(i+1)*50]

        elif mix == 'alternate':
            print('alternate mixing!')
            for i in range(500):
                idx = range(i*50, (i+1)*50, 2)
                X_train[idx] = X_test_old[idx]
                X_test[idx] = X_train_old[idx]
                lipreader_lrw_val_softmax[idx] = lipreader_lrw_test_softmax_old[idx]
                lipreader_lrw_test_softmax[idx] = lipreader_lrw_val_softmax_old[idx]

    return X_train, y_train, X_test, y_test, lipreader_lrw_val_softmax, lipreader_lrw_test_softmax, lrw_correct_one_hot_y_arg


#####################################
# GBT, etc.
#####################################


def compare_assessors(X, y):

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

