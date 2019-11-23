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
    all_feats : bytes
        Contents of text file containing all features and class label.
    pca_stats : bytes
        Contents of text file containing PCA features and class label.

    Returns
    -------
    rf_model : fn
        Trained random forest ensemble.
    pca_model : fn
        PCA transformation.
    """

    coeffs = load_feature_data(all_feature_data,range(2,49))

    pca = decomposition.PCA(n_components=44, whiten=True, svd_solver='auto')
    pca.fit(coeffs)

    #feat_strengths = pca.explained_variance_ratio_
    (training_set, training_classes) = load_training_set(pca_feature_data, range(1,45))

    rf=RandomForestClassifier(n_estimators=1000, max_depth = 4, max_features=2, min_samples_leaf = 4, min_samples_split=2)
    rf.fit(training_set,training_classes)

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

def load_feature_data(feature_data, use_cols):
    """Function to load the feature data from a bytes stream

    Inputs:
        feature_data        bytes       Feature metrics training set
        use_cols            list        List of columns to select from the file

    Outputs:
        data                np.array    Feature numerical data array
    """

    feature_data = ensure_str(feature_data)

    lines = feature_data.split('\n')

    data = []
    for l in lines:
        if len(l) > 0:
            col_data = l.split()
            entry = []
            for col in use_cols:
                try:
                    entry.append(float(col_data[col]))
                except IndexError:
                    pass
            data.append(entry)

    return np.array(data)

def ensure_str(data):
    if type(data) == bytes:
        data = data.decode("utf-8")
    return data

def load_training_set(feature_data, use_cols):
    """Function to load the feature data from a bytes stream

    Inputs:
        feature_data        bytes       Feature metrics training set
        use_cols            list        List of columns to select from the file

    Outputs:
        data                np.array    Feature numerical data array
        training_classes    np.array    Assigned classes
    """

    feature_data = ensure_str(feature_data)

    lines = feature_data.split('\n')

    data = []
    training_classes = []
    for l in lines:
        if len(l) > 0:
            col_data = l.split()
            training_classes.append(col_data[0])
            entry = []
            for col in use_cols:
                try:
                    entry.append(float(col_data[col]))
                except IndexError:
                    pass
            data.append(entry)

    return np.array(data), np.array(training_classes)
