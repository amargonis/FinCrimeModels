import os
import pandas as pd
import teradataml as tdml
from teradataml.context.context import *
    
def trav_tree(node, features, node_count=0):
    node_count = node_count+1
    if node['maxDepth_'] == 0:# this is leaf
        if 'responseCounts_' in node:
            print('leaf with responseCounts_')
            #print(node)
        return features, node_count
    split_feature = node['split_']['attr_']
    score = node['split_']['score_']
    new_value = score if split_feature not in features else features[split_feature]+score
    features.update( {split_feature : new_value} )
    features, node_count = trav_tree(node['rightChild_'], features, node_count)
    features, node_count = trav_tree(node['leftChild_'], features, node_count)
    return features, node_count


def get_global_feature_importance(vantage_model):
    f = {}
    for tree in vantage_model:
        f, n_c = trav_tree(tree,f)
        tree['node_count'] = n_c
    s = {k:v/len(vantage_model) for k,v in f.items()}
    return sorted(s.items(), key = lambda k:k[1], reverse=True)