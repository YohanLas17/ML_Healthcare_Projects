# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold as SKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import pandas as pd
import scipy.stats as stats
from clean_data import NSD


def pred_log(logreg, X_train, y_train, X_test, flag=False):
    """

    :param logreg: An object of the class LogisticRegression
    :param X_train: Training set samples
    :param y_train: Training set labels 
    :param X_test: Testing set samples
    :param flag: A boolean determining whether to return the predicted probabilities of the classes or not (relevant after Q11)
    :return: A two elements tuple containing the predictions and the weighted matrix
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    logreg.fit(X_train, y_train)
    w_logs=logreg.coef_ 
    if not flag:
        y_pred_log=logreg.predict(X_test)
    else:
        y_pred_log = logreg.predict_proba(X_test)
    # -------------------------------------------------------------------------
    return y_pred_log, w_logs


def w_no_p_table(w, features):
    x = np.arange(len(features))
    width = 0.5  # the width of the bars
    mode_name = ['Normal', 'Suspect', 'Pathology']
    fig, axs = plt.subplots(figsize=(20, 10), nrows=3)
    for idx, ax in enumerate(axs):
        ax.bar(x, w[idx, :], width)
        ax.set(xticks=x, xticklabels=features, ylabel='w', title=mode_name[idx])
    fig.tight_layout()
    plt.show()


def w_all_tbl(w2, w1, orig_feat):
    idx_l2 = np.argsort(-w2, axis=1)
    w2_sort = -np.sort(-w2, axis=1)
    w1_sort = np.zeros_like(w2_sort)
    mode_name = ['Normal', 'Suspect', 'Pathology']
    lbl = ['L2', 'L1']
    col = ['orange', 'green']
    feature_dict = {}
    for i in range(w2_sort.shape[0]):
        w1_sort[i, :] = w1[i, idx_l2[i, :]]
        feature_dict[mode_name[i]] = [orig_feat[x] for x in idx_l2[i, :]]
    width = 0.4
    w_tot = [w2_sort, w1_sort]
    fig, axs = plt.subplots(figsize=(20, 10), nrows=3)
    x_orig = np.arange(len(orig_feat))
    x = np.arange(len(orig_feat)) - width/2
    for idx_w, w in enumerate(w_tot):
        for idx_ax, ax in enumerate(axs):
            ax.bar(x, w[idx_ax, :], width, label=lbl[idx_w], color=col[idx_w])
            ax.set(xticks=x_orig, xticklabels=feature_dict[mode_name[idx_ax]], ylabel='w', title=mode_name[idx_ax])
            ax.legend()
        x += width
    fig.tight_layout()
    plt.show()


def cv_kfold(X, y, C, penalty, K, mode):
    """
    
    :param X: Training set samples
    :param y: Training set labels 
    :param C: A list of regularization parameters
    :param penalty: A list of types of norm
    :param K: Number of folds
    :param mode: Mode of normalization (parameter of norm_standard function in clean_data module)
    :return: A list of dictinaries as explained in the notebook
    """
    kf = SKFold(n_splits=K)
    validation_list = []
    scaler = NSD()
    for c in C:
        for p in penalty:
            logreg = LogisticRegression(solver='saga', penalty=p, C=c, max_iter=10000, multi_class='multinomial')
            loss_val_vec = np.zeros(K)
            k = 0
            for train_idx, val_idx in kf.split(X, y):  # val=validation
                x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
        # ------------------ IMPLEMENT YOUR CODE HERE:---------------------------
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
                if mode == 'standard':
                    scaler.fit(x_train)  # Fit only on the training data
                    x_train_nsd = scaler.transform(x_train)  # Transform training data
                    x_val_nsd = scaler.transform(x_val)  # Transform validation data  
               
                y_pred_probs, w_preb = pred_log(logreg, x_train_nsd, y_train, x_val_nsd, flag=True)
                loss_val_vec[k] = log_loss(y_val, y_pred_probs)
               
                k += 1
                
            mu= np.mean(loss_val_vec) # Mean log-loss across K folds
            sigma = np.std(loss_val_vec) # Standard deviation of log-loss across K folds
        
            validation_list.append({'C': c, 'penalty': p, 'mu': mu, 'sigma': sigma})  # Append results to validation_list
        # --------------------------------------------------------------------------
    return validation_list




     
          

