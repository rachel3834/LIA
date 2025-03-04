# -*- coding: utf-8 -*-
"""
    @author: rstreet
"""
import numpy as np
import unittest
from LIA import models
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition

def test_create_models():

    coeffs_data = open('all_features.txt','r').read().encode('utf-8')
    pca_data = open('pca_features.txt','r').read().encode('utf-8')

    (rf, pca) = models.create_models(coeffs_data, pca_data)

    assert type(rf) == type(RandomForestClassifier())
    assert type(pca) == type(decomposition.pca.PCA())
    
if __name__ == '__main__':
    test_create_models()
