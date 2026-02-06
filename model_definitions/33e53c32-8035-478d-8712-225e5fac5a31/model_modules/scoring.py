import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os,json,uuid
from teradataml.dataframe.copy_to import copy_to_sql
from teradataml.context.context import *
from datetime import datetime
from scipy.spatial import distance_matrix
from .teradataFeatureCalculator import featureCalculator as tdFC
from .teradataFinCrimeUtils import fcUtils as fcUtils
from .teradataFinCrimeAlerts import finCrimeAlerts as fcAlerts
from datetime import datetime
from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext
)

def process_metadata(model_version, model_ID, data_conf, anomaly_all_data_table_name, as_of_date ,timestamp):
    local_exp_metadata = pd.DataFrame(
        columns=["model_version", "model_id", "database_name",
                 "anomaly_results_table", "scoring_table", "as_of_date", "job_trigger_timestamp"])

    local_exp_metadata = local_exp_metadata.append({"model_version": model_version,
                                                    "model_id": model_ID,
                                                    "database_name": data_conf["datascience_db"],
                                                    "anomaly_results_table": anomaly_all_data_table_name,
                                                    "scoring_table":data_conf["Score_ADS_name"],
                                                    "as_of_date": as_of_date,
                                                    "job_trigger_timestamp": timestamp
                                                    }, ignore_index=True)

    copy_to_sql(df=local_exp_metadata, table_name="AML_scoring_metadata",
                schema_name=data_conf["datascience_db"],
                temporary=False,
                if_exists='append')
    
    

def convert_date(str_date_time):
    return datetime.strptime(str_date_time, '%d/%m/%Y_%H:%M:%S')

def reconvert_date(dateObj):
    return dateObj.strftime("%d/%m/%Y_%H:%M:%S")


def generate_alert(data_conf, all_alert_ads, crime_type, model_version , model_ID, as_of_date, conn):  # local_exp contains scoring results
    fcAlerts.create_tables(data_conf)
    alert_ads = all_alert_ads[all_alert_ads['Score'] >= data_conf["alert_threshold"]]
    carry_over_custom = fcAlerts.suppress_custom_alert(data_conf, alert_ads, crime_type, model_version , model_ID, as_of_date, conn)
    if not carry_over_custom.empty:
        carry_over_open = fcAlerts.suppress_open_alert(data_conf, carry_over_custom, crime_type, model_version , model_ID, as_of_date, conn)
        if not carry_over_open.empty:
            fcAlerts.suppress_closed_alert(data_conf, carry_over_open, crime_type, model_version , model_ID, as_of_date, conn)   
            
            
def get_priority_level(prob):
    if prob < 0.6:
        return "Low"
    elif prob < 0.75:
        return "Medium"
    elif prob < 0.85:
        return "Medium-High"
    else:
        return "High"
    
def Euclidean_Dist(df1, df2, cols):
    return np.linalg.norm(df1[cols].values - df2[cols].values, axis=1)

def generateAlerts(data_conf, model_conf, **kwargs):
    print("====>Re-Alerting started..")
    hyperparams = model_conf["hyperParameters"]

    # AOA model ID and Version GUID
    modelVersion = kwargs.get("model_version")
    modelId = kwargs.get("model_id")
    
    # connect to the Teradata system
    conn = fcUtils.create_connection(data_conf)
    tdFC.initFC(data_conf["featureStore_db_name"], data_conf["dataScience_db_name"], data_conf["metadata_db_name"] )
    print("step_1")
    
    # Feature Calculator feature set ID and version
    featureSetId = data_conf["featureSetId"]
    featureSetVersion = data_conf["featureSetVersion"]

    # 0
    #
    # Initialize the Alert Pipeline
    
    alertConfig = fcAlerts.finCrimeAlertConfig()
    alertConfig.featureStoreDb = data_conf["featureStore_db_name"]
    alertConfig.dataScienceDb = data_conf["dataScience_db_name"]
    alertConfig.metadataDb = data_conf["metadata_db_name"]
    alertConfig.alertThreshold = hyperparams['alert_threshold']
    alertConfig.recencyThreshold = hyperparams['recency_threshold']
    alertConfig.similarityThreshold = hyperparams['similarity_threshold']
    alertConfig.alertType = "finCrime"
    alertManager = fcAlerts.finCrimeAlertManager(alertConfig, conn)

    scoreingDate = datetime.now().strftime("%Y-%m-%d")
    
    if hyperparams["scoring_date"] == "CURRENT_DATE":
        someDate = datetime.now().strftime("%Y-%m-%d")
    else:
        someDate = datetime.strptime(hyperparams["scoring_date"].split("'")[1], '%Y-%m-%d')
        
    sql = f"""select object_type, object_id, cluster_id, as_of_date, anomaly_score
        from {data_conf["dataScience_db_name"]}.anomaly_results
        where as_of_date = {hyperparams["scoring_date"]} 
            and datascience_model_version = '{modelVersion}' 
            and anomaly_score >= {alertConfig.alertThreshold}
        """
    print(sql)

    dfAlertCandidates = pd.read_sql_query(sql,conn)
   
    print("step_2")
    for index, anomalyRow in dfAlertCandidates.iterrows():
        
        scoredObject = {
            "datascience_model_version" : modelVersion,
            "datascience_model_id" : modelId,  
            "object_type" : anomalyRow['object_type'],
            "object_id" : int(anomalyRow['object_id']),
            "cluster_id" :  anomalyRow['cluster_id'],
            "as_of_date" :  someDate,
            "score_date" :  scoreingDate,
            "anomaly_score" : anomalyRow['anomaly_score']}
        
        alertManager.createAlert(scoredObject)

    alertManager.commit()
    fcUtils.close_connection()


def evaluate(context: ModelContext, **kwargs):

    aoa_create_context()
    data_conf = context.dataset_info.legacy_data_conf
    hyperparams = context.hyperparams


    # AOA model ID and Version GUID
    modelVersion = kwargs.get("model_version")
    modelId = kwargs.get("model_id")
    
    dataSciDb = data_conf["dataScience_db_name"]
    
    # connect to the Teradata system
    conn = fcUtils.create_connection(data_conf)
    
    # create a new dictionary for kv pairs
    scores = {}

    sql1 = f"""select count  (distinct cluster_id) as countClusters 
         from {dataSciDb}.cluster_explainability
         where datascience_model_id ='{modelId}'
         and datascience_model_version = '{modelVersion}'"""
            
    pdfOverview = pd.read_sql(sql1, conn)
    for row in pdfOverview.iterrows():
        scores['Cluster Count'] = str(row[1].get("countClusters"))
            
    sql2 = f"""select cluster_id, count(object_id) as countItems
             from {dataSciDb}.cluster_results
             where datascience_model_id ='{modelId}'
             and datascience_model_version = '{modelVersion}'
             group by cluster_id
             order by cluster_id"""

    pdfClusters = pd.read_sql(sql2, conn)
    for row in pdfClusters.iterrows():
        scores[f"""Cluster {row[1]["cluster_id"]} count"""] = str(row[1]["countItems"])
    
    sql3 = f"""select cluster_id, feature, feature_importance, avg_value
            from {dataSciDb}.cluster_explainability 
            where datascience_model_id ='{modelId}'
            and datascience_model_version = '{modelVersion}'
            order by cluster_id"""
        
    pdfClustersDetails = pd.read_sql(sql3, conn)
    for row in pdfClustersDetails.iterrows():
        scores[f"""Cluster {row[1]["cluster_id"]} feature {row[1]["feature"].replace("_", " ")} importance"""] = str(row[1]["feature_importance"])
        
        scores[f"""Cluster {row[1]["cluster_id"]} feature {row[1]["feature"].replace("_", " ")} centroid"""] = str(row[1]["avg_value"])
        
    print(scores)
    #with open("models/evaluation.json", "w+") as f:
    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(scores, f)
        
    # dump results as json file evaluation.json to models/ folder
    print("Evaluation complete...")
    
    fcUtils.close_connection()

def score(context: ModelContext, db=None, deploy=True, **kwargs):
    print("====>Scoring started..")
    
    aoa_create_context()
    data_conf = context.dataset_info.legacy_data_conf
    hyperparams = context.hyperparams

    #hyperparams = model_conf["hyperParameters"]

    # AOA model ID and Version GUID
    modelVersion = kwargs.get("model_version")
    modelId = kwargs.get("model_id")
    
    # connect to the Teradata system
    conn = fcUtils.create_connection(data_conf)
    tdFC.initFC(data_conf["featureStore_db_name"], data_conf["dataScience_db_name"], data_conf["metadata_db_name"] )
    print("step_1")
    
    # Feature Calculator feature set ID and version
    featureSetId = data_conf["featureSetId"]
    featureSetVersion = data_conf["featureSetVersion"]

    # 0
    #
    # Initialize the Alert Pipeline
    
    alertConfig = fcAlerts.finCrimeAlertConfig()
    alertConfig.featureStoreDb = data_conf["featureStore_db_name"]
    alertConfig.dataScienceDb = data_conf["dataScience_db_name"]
    alertConfig.metadataDb = data_conf["metadata_db_name"]
    alertConfig.alertThreshold = hyperparams['alert_threshold']
    alertConfig.recencyThreshold = hyperparams['recency_threshold']
    alertConfig.similarityThreshold = hyperparams['similarity_threshold']
    alertConfig.alertType = "finCrime"
    alertManager = fcAlerts.finCrimeAlertManager(alertConfig, conn)

    # 1.0
    #
    # Cluster objects that have not already been clustered by first clustering 
    #  all then applying old clustering value back to new parties
    # until I figure out the SQL and fix the rest of the duplicated code.
    # score_clustering == none = do not execute clustering during scoring operation
    # score_clustering == new = cluster only objects that are not already in a cluster
    # score_clustering == all = cluster all objects.  
    # This may result in an object changing clusters if activity has changed
    if hyperparams["score_clustering"] != "none":
        recluster(tdFC, data_conf, featureSetId, featureSetVersion, 
            hyperparams, modelId, modelVersion, conn)

    # 2.0
    #
    # Get the data set from the feature calculator for the scoring operation 
    print("step_2")
    anomaly_features_names,anomaly_features_pos_weights,anomaly_features_neg_weights,objectType = tdFC.getClusteredFeatureWeights(featureSetId, featureSetVersion, conn)


    # 3.0
    #
    # Get the feature weights and the model max score by cluster
    featureWeightDef, clusterMaxScore = tdFC.getModelMaxScore(featureSetId, featureSetVersion, modelVersion, conn)

    # 4.0 
    #
    # Score each object for each feature
    scoredObjects = []
    scoredObject = {}
    scoreingDate = datetime.now().strftime("%Y-%m-%d")
    
    if hyperparams["scoring_date"] == "CURRENT_DATE":
        someDate = datetime.now().strftime("%Y-%m-%d")
    else:
        someDate = datetime.strptime(hyperparams["scoring_date"].split("'")[1], '%Y-%m-%d')

    # Object Scoring        
    columnsList =["datascience_model_version", "datascience_model_id", "object_type", "object_id", 
                  "cluster_id", "as_of_date", "score_date", "anomaly_score"]
    dfScoreResults = pd.DataFrame(columns=columnsList)

    # Feature Scoring
    featsColList =["datascience_model_version", "datascience_model_id", "object_type", "object_id", 
              "as_of_date", "score_date", "feature", "feature_value", "feature_score"]

    dfScoredFeats = pd.DataFrame(columns=featsColList)
    featCount = 0
    objectCount = 0
    print(f"===>Starting model Scoring: {datetime.now()}")

    modelViewName = f"v_modelDefinition_{featureSetId}_{featureSetVersion}"
    anomaly_features_names = anomaly_features_names.split(",")
    for feature_name in anomaly_features_names:
        #feature_name = featureRow['column_name']
        sql = f"""
            insert into {data_conf["dataScience_db_name"]}.anomaly_result_details 
            select '{modelVersion}' as datascience_model_version, 
                '{modelId}' as datascience_model_id, '{objectType}',
                {objectType}, {hyperparams['training_date']}, CURRENT_DATE,
                '{feature_name}' as feature, {feature_name} as feature_value,
                case 
                    when feature_value > avg_value and std_value > 0
                        then  ((feature_value - avg_value) /std_value) * anomaly_pos_weight 
                    when feature_value < avg_value and std_value > 0
                        then abs((feature_value - avg_value) /std_value) * anomaly_neg_weight
                    else 0 
                end as feature_score

            from {data_conf["metadata_db_name"]}.v_model_feature mf ,
                {data_conf["featureStore_db_name"]}.{modelViewName} md
            JOIN {data_conf["dataScience_db_name"]}.v_lastest_object_cluster cr 
                on ( md.{objectType} = cr.object_id )
            join {data_conf["dataScience_db_name"]}.cluster_explainability ce 
                on (ce.cluster_id = cr.cluster_id) 
            where md.fc_agg_summary_date = {hyperparams['training_date']}
                and mf.column_name = '{feature_name}'
                and ce.feature = '{feature_name}'
                and cr.datascience_model_version = '{modelVersion}' 
                and ce.datascience_model_version = '{modelVersion}' 
                and is_anomaly = 1 
                and model_id= {featureSetId} 
                and model_version = {featureSetVersion} 
                """
        print("======")
        print(sql)
        try:
            conn.execute(sql)
        except Exception as inst:
            print("error creating Anomaly Details")
            print(inst)
            
    clusterCount = clusterMaxScore["cluster_count"]
    for i in range(clusterCount):
        print(f"===== for cluster {i} of {clusterCount} clusters ======")
        clusterMin = clusterMaxScore[f"cluster_{i}_min"]
        clusterRange = clusterMaxScore[f"cluster_{i}_max"] - clusterMin
        if clusterRange > 0:
            sql = f"""
            insert into {data_conf["dataScience_db_name"]}.anomaly_results
            select '{modelVersion}' as datascience_model_version, 
                '{modelId}' as datascience_model_id, ard.object_type, 
                ard.object_id, cr.cluster_id, {hyperparams['training_date']}, 
                CURRENT_DATE, (SUM(feature_score) - {clusterMin}) / {clusterRange}
            FROM {data_conf["dataScience_db_name"]}.anomaly_result_details ard
                JOIN {data_conf["dataScience_db_name"]}.v_lastest_object_cluster cr 
                    on ( ard.object_id = cr.object_id  
                        and ard.datascience_model_version = cr.datascience_model_version)
            where ard.as_of_date = {hyperparams['training_date']}
                and ard.score_date = CURRENT_DATE
                and cr.cluster_id = {i}
            group by ard.object_type, ard.object_id, cr.cluster_id
            """
        
            print(sql)
            try:
                conn.execute(sql)
            except Exception as inst:
                print("error creating Anomaly Records")
                print(inst)
        
    fcUtils.close_connection()


def recluster(tdFC, data_conf, featureSetId, featureSetVersion, hyperparams, model_ID, model_version, conn):
    print("====>Reclustering started...")
    # Get a list of columns to cluster and the primary ID column of the feature set (ID = object that is being clustered against)
    cluster_features_names,numeric_feature_names,categoric_feature_names,ID = tdFC.getClusteredFeatures(featureSetId, featureSetVersion, conn)
    
    # Get the data to be clustered
    if hyperparams["score_clustering"] == "new":
        pdfSourceData = tdFC.getUnclusterDataSet(featureSetId, featureSetVersion, hyperparams["training_date"], model_version, conn )
    else:
        pdfSourceData = tdFC.getClusterDataSet(featureSetId, featureSetVersion, hyperparams["training_date"], conn )
        
    if len(pdfSourceData.index) > 0:
        print("reclustering required for new objects")
        # ID and date should not be scaled, remove them fom the result set, scale, then re-add
        IDs_date = pdfSourceData[[ID,"fc_agg_summary_date"]]
        if len(categoric_feature_names) > 0:
            categoric_feature_values = pdfSourceData[categoric_feature_names.split(",")]
        if len(numeric_feature_names) > 0:
            numeric_feature_values = pdfSourceData[numeric_feature_names.split(",")]
            cluster_features_scaled = StandardScaler().fit_transform(numeric_feature_values)
    
        if len(categoric_feature_names) > 0 and len(numeric_feature_names) > 0:
            cluster_ready_data = pd.concat([categoric_feature_values,cluster_features_scaled],axis=1)
        else:
            cluster_ready_data = categoric_feature_values if len(categoric_feature_names) > 0 else cluster_features_scaled
    
        # Get the stored version of the model
        clustered_model = tdFC.getStoredModel(model_version, conn)
   
        print("step_8")
        preds = clustered_model.predict(cluster_ready_data)
        
        if hyperparams["training_date"] == "CURRENT_DATE":
            someDate = datetime.now().strftime("%Y-%m-%d")
        else:
            someDate = datetime.strptime(hyperparams["training_date"].split("'")[1], '%Y-%m-%d')
    
        print("saving new cluster data")
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
        
    # 3.0
    #
    # For each feature, generate the min, max, and std_dev
    viewName = f"""{data_conf["featureStore_db_name"]}.v_modelDefinition_{featureSetId}_{featureSetVersion}"""
    anomaly_features_names,anomaly_features_pos_weights,anomaly_features_neg_weights,ID = tdFC.getClusteredFeatureWeights(featureSetId, featureSetVersion, conn)

    anomaly_features_names = anomaly_features_names.split(",")
    for anom_feature in anomaly_features_names:    
        sql = f"""
            Update A
            from {data_conf["dataScience_db_name"]}.cluster_explainability A, 
            (select '{model_version}' as datascience_model_version, 
                '{model_ID}' as datascience_model_id, cluster_id, 
                '{anom_feature}' as feature, avg({anom_feature}) avg_value, 
                min({anom_feature}) min_value, max({anom_feature}) max_value, 
                stddev_pop({anom_feature}) std_value
            from {viewName} md 
                JOIN {data_conf["dataScience_db_name"]}.cluster_results cr 
                    on (md.{ID} = cr.object_id 
                        and cr.datascience_model_version = '{model_version}'
                        and cr.object_type = '{ID}'
                        and md.fc_agg_summary_date = cr.fc_agg_summary_date
                        and md.fc_agg_summary_date = {hyperparams["training_date"]})
            group by 1,2,3,4) B 
             
            set avg_value = B.avg_value, 
                min_value = B.min_value,
                max_value = B.max_value,
                std_value = B.std_value
 
            where A.cluster_id = B.cluster_id 
                and A.feature = '{anom_feature}'
                and A.datascience_model_version = '{model_version}' 
                and A.datascience_model_id = '{model_ID}'
            """
             
        print(sql)
        try:
            conn.execute(sql)
        except Exception as inst:
            print("""error creating feature value, 
                this can happen when a feature is used for clustering and anomaly 
                detection and can be ignored for duplicate row errors""")
            print(inst)