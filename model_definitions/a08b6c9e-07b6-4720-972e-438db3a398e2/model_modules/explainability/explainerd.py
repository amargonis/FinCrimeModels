#!/usr/bin/env python3
# coding: utf-8
# python3 explainer_train.py --username jd186070 --host tdprd2.td.teradata.com --db-schema ADLDEMO_BustOutDev --table-trees trans_feature_df_model --table-data df_test

import sys

import treelite
import treelite_runtime  # runtime module
import numpy as np
from numpy import column_stack as np_column_stack
import shap

def get_explainer (model_file, data, predictors):
    predictor = treelite_runtime.Predictor(model_file, verbose=True)
    
    def _model_predict_proba(data):
        # Run predict on the output of decision forest. Shape of preds is (data.shape[0],)
        preds = predictor.predict(treelite_runtime.Batch.from_npy2d(data))
        print ('preds', preds)
        # Return preds and complements in right shape
        return np_column_stack([1-preds, preds])
    
    explainer_data = data[predictors].values
    
    explainer = shap.KernelExplainer(_model_predict_proba, explainer_data)
    
    return explainer
    
    
    