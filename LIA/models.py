# -*- coding: utf-8 -*-
"""
    Created on Sat Jan 21 23:59:14 2017

    @author: danielgodinez
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition

def create_models(all_feature_data, pca_feature_data):
    """Creates the Random Forest model and PCA transformation used for classification.

    Parameters
    ----------
    all_feats : str
        Contents of text file containing all features and class label.
    pca_stats : str
        Contents of text file containing PCA features and class label.

    Returns
    -------
    rf_model : fn
        Trained random forest ensemble.
    pca_model : fn
        PCA transformation.
    """
    coeffs = np.genfromtxt(all_feature_data, delimiter=' ', usecols=np.arange(2,49))

    pca = decomposition.PCA(n_components=44, whiten=True, svd_solver='auto')
    pca.fit(coeffs)

    #feat_strengths = pca.explained_variance_ratio_
    training_set = np.genfromtxt(pca_feature_data, delimiter=' ')

    rf=RandomForestClassifier(n_estimators=1000, max_depth = 4, max_features=2, min_samples_leaf = 4, min_samples_split=2)
    rf.fit(training_set[:,np.arange(1,45)].astype(float),training_set[:,0])

    return rf, pca


def create_models_from_files(all_feats, pca_feats):
    """Creates the Random Forest model and PCA transformation used for classification.

    Parameters
    ----------
    all_feats : str
        Name of text file containing all features and class label.
    pca_stats : str
        Name of text file containing PCA features and class label.

    Returns
    -------
    rf_model : fn
        Trained random forest ensemble.
    pca_model : fn
        PCA transformation.
    """
    coeffs = np.loadtxt(all_feats,usecols=np.arange(2,49))
    pca = decomposition.PCA(n_components=44, whiten=True, svd_solver='auto')
    pca.fit(coeffs)
    #feat_strengths = pca.explained_variance_ratio_
    training_set = np.loadtxt(pca_feats, dtype = str)
    rf=RandomForestClassifier(n_estimators=1000, max_depth = 4, max_features=2, min_samples_leaf = 4, min_samples_split=2)
    rf.fit(training_set[:,np.arange(1,45)].astype(float),training_set[:,0])

    return rf, pca
