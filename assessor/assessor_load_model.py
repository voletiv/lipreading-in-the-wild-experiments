import glob

from assessor_model import *
from assessor_train_params import *

######################################################
# MAKE MODEL
######################################################

assessor = my_assessor_model(use_CNN_LSTM=use_CNN_LSTM, use_head_pose=use_head_pose, mouth_nn=mouth_nn, trainable_syncnet=trainable_syncnet,
                             grayscale_images=grayscale_images, my_resnet_repetitions=my_resnet_repetitions,
                             conv_f_1=conv_f_1, conv_f_2=conv_f_2, conv_f_3=conv_f_3, mouth_features_dim=mouth_features_dim,
                             lstm_units_1=lstm_units_1, use_softmax=use_softmax, use_softmax_ratios=use_softmax_ratios,
                             individual_dense=individual_dense, lr_dense_fc=lr_dense_fc, lr_softmax_fc=lr_softmax_fc,
                             dense_fc_1=dense_fc_1, dropout_p1=dropout_p1, dense_fc_2=dense_fc_2, dropout_p2=dropout_p2, last_fc=last_fc)

assessor.load_weights(sorted(glob.glob("*.hdf5"))[0])

assessor.save('assessor.hdf5')

