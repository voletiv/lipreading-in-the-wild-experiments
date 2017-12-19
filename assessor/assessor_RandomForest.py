import numpy as np
import tqdm

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
# RANDOM FOREST
#####################################

# PCA
pca_dense = decomposition.PCA(n_components=32)
pca_dense.fit(lipreader_lrw_val_dense)
lipreader_lrw_val_dense = pca_dense.transform(lipreader_lrw_val_dense)
lipreader_lrw_test_dense = pca_dense.transform(lipreader_lrw_test_dense)
pca_softmax = decomposition.PCA(n_components=32)
pca_softmax.fit(lipreader_lrw_val_softmax)
lipreader_lrw_val_softmax = pca_softmax.transform(lipreader_lrw_val_softmax)
lipreader_lrw_test_softmax = pca_softmax.transform(lipreader_lrw_test_softmax)

# n_frames, Dense, Softmax
train_features = np.hstack((np.reshape(lrw_val_n_of_frames, (len(lrw_val_n_of_frames), 1)), lipreader_lrw_val_dense, lipreader_lrw_val_softmax))
train_y = lipreader_lrw_val_correct_or_wrong
X_test = np.hstack((np.reshape(lrw_test_n_of_frames, (len(lrw_test_n_of_frames), 1)), lipreader_lrw_test_dense, lipreader_lrw_test_softmax))
y_test = lipreader_lrw_test_correct_or_wrong

# n_frames, Dense, Softmax
train_features = np.hstack((np.reshape(lrw_val_n_of_frames, (len(lrw_val_n_of_frames), 1)), lipreader_lrw_val_dense, lipreader_lrw_val_softmax))
train_y = lipreader_lrw_val_correct_or_wrong
X_test = np.hstack((np.reshape(lrw_test_n_of_frames, (len(lrw_test_n_of_frames), 1)), lipreader_lrw_test_dense, lipreader_lrw_test_softmax))
y_test = lipreader_lrw_test_correct_or_wrong

# n_frames, Dense
train_features = np.hstack((np.reshape(lrw_val_n_of_frames, (len(lrw_val_n_of_frames), 1)), lipreader_lrw_val_dense))
train_y = lipreader_lrw_val_correct_or_wrong
X_test = np.hstack((np.reshape(lrw_test_n_of_frames, (len(lrw_test_n_of_frames), 1)), lipreader_lrw_test_dense))
y_test = lipreader_lrw_test_correct_or_wrong

# Default
X_train = train_features
y_train = train_y

# Split train into train and val
X_train, X_val, y_train, y_val = train_test_split(train_features, train_y, test_size=0.33, random_state=0)

# Mix train and test
X_train, X_test, y_train, y_test = train_test_split(np.vstack((train_features, X_test)), np.append(train_y, y_test), test_size=0.33, random_state=0)

# Choose 50-50
pos_X = train_features[np.where(lipreader_lrw_val_correct_or_wrong == 1)]
pos_y = lipreader_lrw_val_correct_or_wrong[np.where(lipreader_lrw_val_correct_or_wrong == 1)]
neg_X = train_features[np.where(lipreader_lrw_val_correct_or_wrong == 0)]
neg_y = lipreader_lrw_val_correct_or_wrong[np.where(lipreader_lrw_val_correct_or_wrong == 0)]
train_features_eq = np.vstack((pos_X[:len(neg_X)], neg_X))
train_y_eq = np.append(pos_y[:len(neg_X)], neg_y)
X_train, X_val, y_train, y_val = train_test_split(train_features_eq, train_y_eq, test_size=0.33, random_state=0)

# Full
X_train = np.vstack((train_features, X_test))
y_train = np.append(train_y, y_test)

#####################################
# TRAIN AND TEST
#####################################

# Random Forest
clf = RandomForestClassifier(n_estimators=10, random_state=0)

# kf = KFold(n_splits=10)
# kf.get_n_splits(X)
kf = KFold(n_splits=10)
# gkf.get_n_splits(X_train, y_train)
for train_index, test_index in tqdm.tqdm(kf.split(X_train, y_train)):
    clf.fit(X_train[train_index], y_train[train_index])

# clf.fit(X_train, y_train)
# clf.fit(X_test, y_test)

clf.score(X_train, y_train)
clf.score(X_val, y_val)
clf.score(X_test, y_test)

