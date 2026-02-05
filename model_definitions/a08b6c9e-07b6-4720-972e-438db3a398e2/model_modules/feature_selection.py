import os
import pandas as pd
import teradataml as tdml
from teradataml.context.context import *

def build_feat_str(r):
    """
    To generate a sql statement for provided feature
    
    :param r: (series): feature name
    """
    st=r.feature
    if r.source=='party': return 'pty'+'.'+st+' as feat'+str(r.feature_id)
    else: return 'agg'+'.'+st+' as feat'+str(r.feature_id)
    #return r.agg_level + '.' + r.feature + 'as feat' + str (r.feature_id)
    
def select_features (data_conf, eng):
    """
    Join features information based on selected model features
    
    :param data_conf: (dict): dataset metadata
    :param eng: (object): Database engine
    """
    model_features = pd.read_sql_query(f"sel * from {data_conf['features_db']}.{data_conf['features_to_featureset_tbl']}", eng)
    feature_metadata = pd.read_sql_query(f"sel * from {data_conf['features_db']}.{data_conf['feature_metadata_tbl']}", eng)
    
    features = model_features[model_features.model_id==data_conf['featureset_id']].merge(feature_metadata, on='feature_id')

    return features
    
def numerical_features (features):
    """
    # prepare a concatenated sql query string for numerical features
    
    :param features: (dataframe): all features
    """
    num_inps = ''
    num_feat_str = ''
    if not features[(features.status=='feat')&(features.ds_type=='n')].empty:
        num_feat_str = '\n,'.join(features[(features.status=='feat')&(features.ds_type=='n')][['feature', 'source', 'feature_id']].apply(build_feat_str, axis=1))
    
        for i in num_feat_str.split('\n,'):
            num_inps+="'"+i.split(' as ')[1]+"',"
        num_inps = num_inps.rstrip(",") 
        num_feat_str = num_feat_str + '\n,'
        
    return num_feat_str, num_inps

def categorical_features (features):
    """
    # prepare a concatenated sql query string for categorical features
    
    :param features: (dataframe): all features
    """
    cat_inps = ''
    cat_feat_str = ''
    if not features[(features.status=='feat')&(features.ds_type=='c')].empty:
        cat_feat_str = '\n,'.join(features[(features.status=='feat')&(features.ds_type=='c')][['feature', 'source', 'feature_id']].apply(build_feat_str, axis=1))
        
        
        for i in cat_feat_str.split('\n,'):
            cat_inps+="'"+i.split(' as ')[1]+"',"
        cat_inps = cat_inps.rstrip(",")
        cat_feat_str = cat_feat_str + '\n,'
        
    return cat_feat_str, cat_inps

def get_id_feature (features):
    """
    # select instance id column
    
    :param features: (dataframe): all features
    """
    inst_id_feat = features[features.status=='id'].feature.values[0]
    return inst_id_feat

def get_label (features):
    """
    # select label column
    
    :param features: (dataframe): all features
    """
    label_feat = features[features.status=='label'].feature.values[0]
    return label_feat