import json
import sys
import pandas as pd
import numpy as np
import pickle,os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from teradataml.dataframe.copy_to import copy_to_sql
from teradataml.context.context import *
from sklearn.ensemble import RandomForestClassifier
from .teradataFeatureCalculator import featureCalculator as tdFC
from .teradataFinCrimeUtils import fcUtils as fcUtils
from datetime import datetime
from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext
)

def delete_record_if_exists(db_name, model_version, conn):
    conn.execute(f"""delete from {db_name}.models_artifacts WHERE model_version='{model_version}'""" )

def loadModel(data_conf, clustered_model, featureSetVersion, featureSetId, model_version, model_ID, conn):

    clustered_model_bytes = pickle.dumps(clustered_model)
    conn.execute(
        f"""insert into {data_conf["dataScience_db_name"]}.model_artifacts (model_version, model_id, model, featureset_version, featureset_id) values(?,?,?,?,?)""",
        [str(model_version), str(model_ID), clustered_model_bytes, str(featureSetVersion), str(featureSetId)])

def cluster_explainability(conn, anomaly_features_names, ID, data, clusterDate, no_of_clusters, centroids, avg_values, min_values, max_values, std_values, model_version, model_ID, dataset_version, featureSetId, data_conf):
    print("=== Cluster Explainability ===")
    print(anomaly_features_names);
    
    models = {}
    columnsList =["datascience_model_version", "datascience_model_id", "cluster_id","feature", "feature_importance", "scaled_centroid", "avg_value", "min_value", "max_value", "std_value"]
    g_exp = pd.DataFrame(columns=columnsList)
    
    # 1.0
    #
    # using the clustered data, train a predictive model to predict the cluster that has already been applied
    #  then, use standard explainability on the predictive model to determine the feature importance 
    #  for predicting the cluster.
    for i in range(no_of_clusters):
        positive_cluster = data[data["cluster_id"] == i]
        negative_cluster = data[data["cluster_id"] != i]
        positive_cluster["cluster_id"] = positive_cluster["cluster_id"].apply(lambda x: 1)
        negative_cluster["cluster_id"] = negative_cluster["cluster_id"].apply(lambda x: 0)

        frames = [positive_cluster, negative_cluster]

        clustered_data = pd.concat(frames)

        X = clustered_data.drop({"cluster_id"}, axis=1)
        #X = clustered_data.drop({"cluster_id"}, axis=1)
        y = clustered_data["cluster_id"]

        model = RandomForestClassifier()
        model.fit(X, y)
        models[i] = model
        feats = {}  # a dict to hold feature_name: feature_importance
       
        for feature, importance, centroid in zip(X.columns, model.feature_importances_, centroids.iloc[i]):
            # feats[feature] = importance
            # feats[feature] = i
            # feats[feature] = centroids.iloc[i].values
            # #feats.update(centroids.iloc[i])
            
            g_exp.loc[len(g_exp.index)] = [model_version, model_ID,i,feature,importance,centroid,avg_values.iloc[i].get(feature),float(min_values.iloc[i].get(feature)),float(max_values.iloc[i].get(feature)),float(std_values.iloc[i].get(feature))]
                               
            #g_exp = g_exp.append({"datascience_model_version": model_version,
            #                      "datascience_model_id": model_ID,
            #                      "cluster_id": i,
            #                      "feature": feature,
            #                      "feature_importance": importance,
            #                      "scaled_centroid": centroid,
            #                      "avg_value": avg_values.iloc[i].get(feature),
            #                      "min_value": float(min_values.iloc[i].get(feature)),
            #                      "max_value": float(max_values.iloc[i].get(feature)),
            #                      "std_value": float(std_values.iloc[i].get(feature)),
            #                      } , ignore_index=True)
    
    try:
        copy_to_sql(g_exp,
                table_name="cluster_explainability", if_exists="append", 
                schema_name=data_conf["dataScience_db_name"])
    except Exception as inst:
        print("error creating explainability")
        print(inst)
        
     
    # 2.0
    #
    # Use Vantage Analytics Library to build a histogram of the features

    viewName = f"""{data_conf["featureStore_db_name"]}.v_modelDefinition_{featureSetId}_{dataset_version}"""

    sql = f"""replace view {data_conf['dataScience_db_name']}.v_featureCluster as 
        Select datascience_model_version, cluster_id, mvd.* from 
        {viewName} mvd join {data_conf['dataScience_db_name']}.cluster_results cr 
            on ({ID} = cr.object_id and cr.object_type = '{ID}')""" 

    try:
        conn.execute(sql)
    except Exception as inst:
        print("error creating clustered feature view")
        print(inst)
        
        
    for i in range(no_of_clusters):
        sql = f"""call {data_conf["analyticsLibrary_db_name"]}.td_analyze('histogram',
            'database={data_conf['dataScience_db_name']}; 
            tablename=v_featureCluster; 
            columns={anomaly_features_names};
            outputdatabase={data_conf["dataScience_db_name"]};
            outputtablename=feature_distribution_temp;
            overwrite=true;
            where=datascience_model_version=''{model_version}'' and cluster_id={i}
            ')
            """
        try:
            conn.execute(sql)
        except Exception as inst:
            print("error running td_analyze")
            print(inst)
            
        sql = f"""insert into {data_conf["dataScience_db_name"]}.cluster_distribution 
            select '{model_version}', {i}, xcol, xbin, xbeg, xend, xcnt, xpct 
            from {data_conf["dataScience_db_name"]}.feature_distribution_temp;
            """
        try:
            conn.execute(sql)
        except Exception as inst:
            print("error inserting cluster_distribution")
            print(inst)



    # 3.0
    #
    # For each feature, generate the min, max, and std_dev
    sql = f"""insert into {data_conf["dataScience_db_name"]}.cluster_explainability_history ("""
    
    anomaly_features_names = anomaly_features_names.split(",")
    for anom_feature in anomaly_features_names:    
        sql = f"""insert into {data_conf["dataScience_db_name"]}.cluster_explainability (
            datascience_model_version, datascience_model_id, cluster_id, feature, avg_value, min_value, max_value, std_value )
            select '{model_version}', '{model_ID}', cluster_id, '{anom_feature}', avg({anom_feature}), min({anom_feature}), max({anom_feature}), stddev_pop({anom_feature})
            from {viewName} md JOIN
            {data_conf["dataScience_db_name"]}.cluster_results cr on (md.{ID} = cr.object_id 
            and cr.object_type = '{ID}'
            and md.fc_agg_summary_date = cr.fc_agg_summary_date
            and md.fc_agg_summary_date = {clusterDate})
            group by cluster_id"""
            
        print(sql)
        try:
            conn.execute(sql)
        except Exception as inst:
            print("error creating feature value, this can happen when a feature is used for clustering and anomaly detection and can be ignored for duplicate row errors")
            print(inst)
     

def train(data_conf, model_conf, **kwargs):
	#def train(context: ModelContext, **kwargs):
    print("=====BEGIN: train AML model")
    
    #aoa_create_context()
    #data_conf = context.dataset_info.legacy_data_conf
    #hyperparams = context.hyperparams
    
    
    hyperparams = model_conf["hyperParameters"]
    conn = fcUtils.create_connection(data_conf)
    tdFC.initFC(data_conf["featureStore_db_name"], data_conf["dataScience_db_name"], data_conf["metadata_db_name"] )
    print("step_1")

    # AOA model ID and Version GUID
    model_version = kwargs.get("model_version")
    model_ID = kwargs.get("model_id")
    
    # Feature Calculator feature set ID and version
    featureSetId = data_conf["featureSetId"]
    featureSetVersion = data_conf["featureSetVersion"]

    # Get a list of columns to cluster and the primary ID column of the feature set (ID = object that is being clustered against)
    cluster_features_names,numeric_feature_names,categoric_feature_names,ID = tdFC.getClusteredFeatures(featureSetId, featureSetVersion, conn)
        
    anomaly_features_names,anomaly_features_pos_weights,anomaly_features_neg_weights,ID = tdFC.getClusteredFeatureWeights(featureSetId, featureSetVersion, conn)
    print("step_2")

    # get the data set to perform clustering
    pdfSourceData = tdFC.getClusterDataSet(featureSetId, featureSetVersion, hyperparams["training_date"], conn )
    print("step_3")
    
    # ID and date should not be scaled, remove them fom the result set, scale, then re-add
    IDs_date = pdfSourceData[[ID,"fc_agg_summary_date"]]
    if len(categoric_feature_names) > 0:
        categoric_feature_values = pdfSourceData[categoric_feature_names.split(",")]
    if len(numeric_feature_names) > 0:
        numeric_feature_values = pdfSourceData[numeric_feature_names.split(",")]
        cluster_features_scaled = StandardScaler().fit_transform(numeric_feature_values)
    
    # pdfSourceData.drop([ID,"fc_agg_summary_date"],inplace=True,axis=1)
    # pdfSourceData.fillna(0,inplace=True) -- FEature calc replaces null with 0, don't need this
    
    if len(categoric_feature_names) > 0 and len(numeric_feature_names) > 0:
        cluster_ready_data = pd.concat([categoric_feature_values,cluster_features_scaled],axis=1)
    else:
        cluster_ready_data = categoric_feature_values if len(categoric_feature_names) > 0 else cluster_features_scaled


    # Run clustering score to determine the "optimal" number of clusters
    clusterCount = int(hyperparams["max_clusters"])
    if hyperparams["cluster_mode"] == "auto":
        km_silhouette = []
        maximum = 0
        clusterCount = 1
        for i in range(2,int(hyperparams["max_clusters"])):
            km = KMeans(n_clusters=i,init=hyperparams["init"], max_iter=hyperparams["max_iter"], n_init=hyperparams["n_init"]).fit(cluster_ready_data)
            preds = km.predict(cluster_ready_data)
            silhouette = silhouette_score(cluster_ready_data,preds)
            if silhouette > maximum:
                maximum = silhouette
                clusterCount = i
            km_silhouette.append(silhouette)
            print(f"Silhouette score for number of cluster(s) {i}: {silhouette}")

    print(f"step_4:executing model with {clusterCount} number of clusters")
    clustered_model = KMeans(n_clusters=clusterCount,init=hyperparams["init"], max_iter=int(hyperparams["max_iter"]), n_init=int(hyperparams["n_init"])).fit(cluster_ready_data)
    
    centroids = pd.DataFrame(clustered_model.cluster_centers_, columns=list(map(lambda x: str(x) + "_centroid", cluster_features_names.split(","))))
    print("step_6")
    loadModel(data_conf, clustered_model, featureSetVersion, featureSetId, model_version, model_ID, conn)
    #print("step_7")
    #cluster_all_data = pd.concat([IDs_date,cluster_ready_data],axis=1)
    print("step_8")
    preds = clustered_model.predict(cluster_ready_data)
    print(preds)
    print("step_9 ")
    
    if hyperparams["training_date"] == "CURRENT_DATE":
        someDate = datetime.now().strftime("%Y-%m-%d")
        print(f"step_9_1 {someDate}")
    else:
        someDate = datetime.strptime(hyperparams["training_date"].split("'")[1], '%Y-%m-%d')
        print(f"""hyperparameter is {hyperparams["training_date"].split("'")[1]}""")
        print(f"step_9_2 {someDate}")
     
    cluster_scores = {}
    cluster_scores['datascience_model_version'] = model_version
    cluster_scores['datascience_model_id'] = model_ID
    cluster_scores['object_type'] = ID
    cluster_scores['object_id'] = IDs_date[ID]
    cluster_scores['fc_agg_summary_date'] = someDate
    cluster_scores['score_date'] = datetime.now().strftime("%Y-%m-%d")
    cluster_scores['cluster_id'] = preds
    pdfClusterResults = pd.DataFrame(cluster_scores) 
    # print(pdfClusterResults)
    
    copy_to_sql(pdfClusterResults, table_name="cluster_results", if_exists="append",
                schema_name=data_conf["dataScience_db_name"])
    
    pdfSourceData.drop([ID,"fc_agg_summary_date"], inplace=True, axis=1)
    pdfSourceData['cluster_id'] = preds
    gb = pdfSourceData.groupby(['cluster_id'])
    avg_values = gb.mean()
    min_values = gb.min()
    max_values = gb.max()
    std_dev = gb.std()
    
    cluster_explainability(conn, anomaly_features_names, ID, pdfSourceData, hyperparams["training_date"], clusterCount, centroids, avg_values, min_values, max_values, std_dev, str(model_version), str(model_ID), str(featureSetVersion), str(featureSetId), data_conf)

    fcUtils.close_connection()