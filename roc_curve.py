# Only need to change train_set_? in line 44
# x for xception
# c for cnn_rnn

import matplotlib.pyplot as plt
import json
from sklearn.metrics import roc_curve, auc
import numpy as np

train_sets_x = ['xception_df', 'xception_f2f', 'xception_fs', 'xception_nt']
train_sets_c = ['cnn_rnn_df', 'cnn_rnn_f2f', 'cnn_rnn_fs', 'cnn_rnn_nt']
# curve_name = 'ROC curve for Resnet-152+LSTM trained on Ori_NT_Train'
test_set = ['df', 'f2f', 'fs', 'nt']


def plot_curve(train_set, curve_name, y_df_labels, y_df_pred, y_f2f_labels, y_f2f_pred, y_fs_labels, y_fs_pred, y_nt_labels, y_nt_pred):
    fpr_df, tpr_df, th_df = roc_curve(y_df_labels, y_df_pred)
    auc_value_df = auc(fpr_df, tpr_df)

    fpr_f2f, tpr_f2f, th_f2f = roc_curve(y_f2f_labels, y_f2f_pred)
    auc_value_f2f = auc(fpr_f2f, tpr_f2f)

    fpr_fs, tpr_fs, th_fs = roc_curve(y_fs_labels, y_fs_pred)
    auc_value_fs = auc(fpr_fs, tpr_fs)

    fpr_nt, tpr_nt, th_nt = roc_curve(y_nt_labels, y_nt_pred)
    auc_value_nt = auc(fpr_nt, tpr_nt)

    plt.plot([0, 1], [0, 1], 'm--')
    plt.plot(fpr_df, tpr_df, 'orange', label='Ori_DF_test')
    plt.plot(fpr_f2f, tpr_f2f, 'blue', label='Ori_F2F_test')
    plt.plot(fpr_fs, tpr_fs, 'red', label='Ori_FS_test')
    plt.plot(fpr_nt, tpr_nt, 'green', label='Ori_NT_test')
    # plt.plot(fpr_df, tpr_df, 'orange', label='Ori_DF_Test AUC= %0.3f' % auc_value_df)
    # plt.plot(fpr_f2f, tpr_f2f, 'blue', label='Ori_F2F_Test AUC = %0.3f' % auc_value_f2f)
    # plt.plot(fpr_fs, tpr_fs, 'red', label='Ori_FS_Test AUC = %0.3f' % auc_value_fs)
    # plt.plot(fpr_nt, tpr_nt, 'green', label='Ori_NT_Test AUC = %0.3f' % auc_value_nt)
    plt.legend(loc=4, prop={'size': 14})
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False positive rate', fontsize=15)
    plt.ylabel('True positive rate', fontsize=15)
    plt.title(curve_name, fontsize=15)
    plt.savefig(train_set+'_roc_curve.png', bbox_inches='tight')
    plt.clf()

for train_set in train_sets_c:
    if train_set == 'xception_df':
        curve_name = 'ROC curve for Xception trained on Ori_DF_train'
    elif train_set == 'xception_f2f':
        curve_name = 'ROC curve for Xception trained on Ori_F2F_train'
    elif train_set == 'xception_fs':
        curve_name = 'ROC curve for Xception trained on Ori_FS_train'
    elif train_set == 'xception_nt':
        curve_name = 'ROC curve for Xception trained on Ori_NT_train'
    elif train_set == 'cnn_rnn_df':
        curve_name = 'ROC curve for ResNet+LSTM trained on Ori_DF_train'
    elif train_set == 'cnn_rnn_f2f':
        curve_name = 'ROC curve for ResNet+LSTM trained on Ori_F2F_train'
    elif train_set == 'cnn_rnn_fs':
        curve_name = 'ROC curve for ResNet+LSTM trained on Ori_FS_train'
    elif train_set == 'cnn_rnn_nt':
        curve_name = 'ROC curve for ResNet+LSTM trained on Ori_NT_train'

    for name in test_set:
        if name == 'df':
            with open(train_set+'_' + name + '_c23_labels.txt', 'r') as f:
                y_df_labels = json.load(f)
                y_df_labels = list(map(int, y_df_labels))
            with open(train_set+'_' + name + '_c23_prediction.txt', 'r') as f:
                y_df_pred = json.load(f)
                y_df_pred = list(map(float, y_df_pred))

        elif name == 'f2f':
            with open(train_set+'_' + name + '_c23_labels.txt', 'r') as f:
                y_f2f_labels = json.load(f)
                y_f2f_labels = list(map(int, y_f2f_labels))
            with open(train_set+'_' + name + '_c23_prediction.txt', 'r') as f:
                y_f2f_pred = json.load(f)
                y_f2f_pred = list(map(float, y_f2f_pred))

        elif name == 'fs':
            with open(train_set+'_' + name + '_c23_labels.txt', 'r') as f:
                y_fs_labels = json.load(f)
                y_fs_labels = list(map(int, y_fs_labels))
            with open(train_set+'_' + name + '_c23_prediction.txt', 'r') as f:
                y_fs_pred = json.load(f)
                y_fs_pred = list(map(float, y_fs_pred))

        elif name == 'nt':
            with open(train_set+'_' + name + '_c23_labels.txt', 'r') as f:
                y_nt_labels = json.load(f)
                y_nt_labels = list(map(int, y_nt_labels))
            with open(train_set+'_' + name + '_c23_prediction.txt', 'r') as f:
                y_nt_pred = json.load(f)
                y_nt_pred = list(map(float, y_nt_pred))

    plot_curve(train_set, curve_name, y_df_labels, y_df_pred, y_f2f_labels, y_f2f_pred, y_fs_labels, y_fs_pred, y_nt_labels, y_nt_pred)
