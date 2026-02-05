#!/usr/bin/env python3
# coding: utf-8
# python3 explainer_train.py --username jd186070 --host tdprd2.td.teradata.com --db-schema ADLDEMO_BustOutDev --table-trees trans_feature_df_model --table-data df_test

import sys

import treelite


def process_leaf_node(treelite_tree, node):
    # Get counts for each label (+/-) at this leaf node
    # TODO Need to test to see if we have mixed samples in the leaf
    fraction_positive = 1 if node['label_']=='1' else 0
    # The fraction above is now the leaf output
    node_id = node['id_']-1 # Vantage node ids starts from 1
    treelite_tree[node_id].set_leaf_node(fraction_positive)
    return treelite_tree

def process_test_node(node, features, treelite_tree, encoders):
    # Initialize the test node with given node ID
    node_id = node['id_']-1 # Vantage node ids starts from 1
    node_feature_id = features.index(node['split_']['attr_'])
    left_child_node_id = node['leftChild_']['id_']-1
    right_child_node_id = node['rightChild_']['id_']-1

    # Check Test Type
    if node['split_']['type_']=='CLASSIFICATION_CATEGORICAL_SPLIT':
        left_cats = list(encoders[node['split_']['attr_']].transform(node['split_']['leftCategories_']))
        treelite_tree[node_id].set_categorical_test_node(
                            feature_id=node_feature_id,
                            left_categories=left_cats,
                            default_left=True,
                            left_child_key=left_child_node_id,
                            right_child_key=right_child_node_id)
    else: #CLASSIFICATION_NUMERICAL_SPLIT
        node_threshold = node['split_']['splitValue_']
        treelite_tree[node_id].set_numerical_test_node(
                            feature_id=node_feature_id,
                            opname='<=',# need to check the operator
                            threshold=node_threshold,
                            default_left=True,
                            left_child_key=left_child_node_id,
                            right_child_key=right_child_node_id)
    return treelite_tree


def process_tree(node, features, treelite_tree, encoders):
    if node['maxDepth_'] == 0:# this is leaf
        treelite_tree = process_leaf_node(treelite_tree, node)
        return treelite_tree

    treelite_tree = process_test_node(node, features, treelite_tree, encoders)

    treelite_tree = process_tree(node['leftChild_'], features, treelite_tree, encoders) #Process left tree
    treelite_tree = process_tree(node['rightChild_'], features, treelite_tree, encoders) #Process right tree

    return treelite_tree


# Features is a ordered list of features used in training the model
def process_model(vantage_model, features, encoders):
    builder = treelite.ModelBuilder(num_feature=len(features), random_forest=True)
    for i in range(len(vantage_model)): # For every tree
        # Process i-th tree and add to the builder
        treelite_tree = treelite.ModelBuilder.Tree()
        # Node #0 is always root for Vantage decision trees
        treelite_tree[0].set_root()
        builder.append( process_tree(vantage_model[i], features, treelite_tree, encoders) )
    return builder.commit()