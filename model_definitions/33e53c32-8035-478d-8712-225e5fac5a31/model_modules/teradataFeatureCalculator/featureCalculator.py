from datetime import datetime
import json, pickle
import pandas as pd
from . import storage

def __init__(self):
    self.featureStore_db_name = ""
    self.dataScience_db_name = ""
    self.metadata_db_name = ""

def initFC(featureStoreDb, dataScienceDb, metadataDb):
    """featureCalculator initiation requires name of featureStoreDB, dataScienceDB, and metadataDb
    featureStoreDB is the database containing the feature calculator _agg tables
    dataScienceDB is the database where temporary or data science objects should be created
    featureCalculatorDB is the database where the feature calc metadata is stored
    """
    print("====>initFC")
    storage.featureStore_db_name = featureStoreDb
    storage.dataScience_db_name = dataScienceDb
    storage.metadata_db_name = metadataDb
    
    

def getFeatureSet(featureSetId, featureSetVersion, conn):
    """Returns a Pandas DataFrame with the features for clustering and the target ID for scoring
        featureSetId: Feature Calculator feature set id
        featureSetVersion: Feature Calculator feature set version
        conn: database connection
    """
    print(f"====>model_features {storage.featureStore_db_name}, {storage.dataScience_db_name}, {storage.metadata_db_name}")
    meta_data = pd.read_sql(f"""select model_version,model_id,column_name, is_cluster,ml_type 
        from {storage.metadata_db_name}.v_model_feature 
        where model_id={featureSetId} and model_version = {featureSetVersion}""" ,conn)
    #features_names = ",".join(meta_data[(meta_data['ml_type'] == 2) & (meta_data['is_cluster'] == 1)]['column_name'].tolist())
    
    ID = meta_data[meta_data['ml_type'] == 1]['column_name'].values[0]
    #return features_names,ID
    return meta_data,ID

def getClusteredFeatures( featureSetId, featureSetVersion, conn):
    """Returns a Pandas DataFrame with the features for clustering and the target ID for scoring
        featureSetId: Feature Calculator feature set id
        featureSetVersion: Feature Calculator feature set version
        conn: database connection
    """
    print(f"====>model_features {storage.featureStore_db_name}, {storage.dataScience_db_name}, {storage.metadata_db_name}")
    meta_data = pd.read_sql(f"""select model_version,model_id,column_name, is_cluster,ml_type, ds_type 
        from {storage.metadata_db_name}.v_model_feature 
        where model_id={featureSetId} and model_version = {featureSetVersion}""" ,conn)
    # Get a list of all features that are to be clustered
    cluster_features_names = ",".join(meta_data[(meta_data['ml_type'] == 2) & (meta_data['is_cluster'] == 1)]['column_name'].tolist())
    # Get a list of cluster features that are numeric 
    numeric_feature_names = ",".join(meta_data[(meta_data['ml_type'] == 2) & (meta_data['ds_type'] == 1) & (meta_data['is_cluster'] == 1)]['column_name'].tolist())
    # Get a list of cluster features that are categorical 
    categoric_feature_names = ",".join(meta_data[(meta_data['ml_type'] == 2) & (meta_data['ds_type'] == 2) & (meta_data['is_cluster'] == 1)]['column_name'].tolist())

    print(cluster_features_names)
    ID = meta_data[meta_data['ml_type'] == 1]['column_name'].values[0]
    return cluster_features_names,numeric_feature_names,categoric_feature_names,ID
   
def getClusterDataSet(featureSetId, featureSetVersion, trainingDate, conn):
    """Return a dataframe with the rows/columns needed for clustering the data
        trainingDate: The feature calculator as_of_date for feature values
    """ 
    cluster_features_names,numeric_feature_names,categoric_feature_names, ID = getClusteredFeatures(featureSetId, featureSetVersion, conn)

    sqlStmt = f"""select {ID},fc_agg_summary_date,{cluster_features_names} 
        from {storage.featureStore_db_name}.v_modelDefinition_{featureSetId}_{featureSetVersion} 
        where fc_agg_summary_date={trainingDate}"""
    print(sqlStmt)
    X = pd.read_sql_query(sqlStmt,conn)
    return X

def getUnclusterDataSet(featureSetId, featureSetVersion, trainingDate, model_version, conn):
    """Return a dataframe with the rows/columns needed for clustering the data
        trainingDate: The feature calculator as_of_date for feature values
    """ 
    cluster_features_names,numeric_feature_names,categoric_feature_names, ID = getClusteredFeatures(featureSetId, featureSetVersion, conn)

    sqlStmt = f"""select {ID},fc_agg_summary_date,{cluster_features_names} 
        from {storage.featureStore_db_name}.v_modelDefinition_{featureSetId}_{featureSetVersion} 
        where fc_agg_summary_date={trainingDate} and 
            {ID} not in (select object_id 
                from {storage.dataScience_db_name}.cluster_results 
                where object_type = '{ID}' and datascience_model_version = '{model_version}')"""
                
    print(sqlStmt)
    X = pd.read_sql_query(sqlStmt,conn)
    return X
def getDataSet(featureSetId, featureSetVersion, scoringDate, conn):
    """Return a dataframe with the rows/columns needed for anomaly detection
        scoringDate: The feature calculator as_of_date for feature values
    """ 
    
    featureSetNames,ID = getFeatureSetNames(featureSetId, featureSetVersion, conn)
    
    columnNames = ",".join(featureSetNames['column_name'].tolist())
    sqlStmt = f"""select fc_agg_summary_date,{columnNames} 
        from {storage.featureStore_db_name}.v_modelDefinition_{featureSetId}_{featureSetVersion} 
        where fc_agg_summary_date={scoringDate}"""
    print(sqlStmt)
    X = pd.read_sql_query(sqlStmt,conn)
    return X

def getClusteredFeatureWeights(featureSetId, featureSetVersion, conn):
    print("====>model_feature weights for anomaly scoring")
    print("step_1")
    meta_data = pd.read_sql(f"""select feature, column_name,is_cluster,is_anomaly,anomaly_pos_weight,anomaly_neg_weight,ml_type 
        from {storage.metadata_db_name}.v_model_feature 
        where model_id={featureSetId} and model_version = {featureSetVersion}""" ,conn)
    print("step_2")
    anomaly_features_names = ",".join(meta_data[(meta_data['ml_type'] == 2) & (meta_data['is_anomaly'] == 1)]['column_name'].tolist())
    anomaly_features_pos_weights = meta_data[(meta_data['ml_type'] == 2) & (meta_data['is_anomaly'] == 1)]['anomaly_pos_weight']
    anomaly_features_neg_weights = meta_data[(meta_data['ml_type'] == 2) & (meta_data['is_anomaly'] == 1)]['anomaly_neg_weight']
    ID = meta_data[meta_data['ml_type'] == 1]['column_name'].values[0]

    return anomaly_features_names,anomaly_features_pos_weights,anomaly_features_neg_weights,ID

def getFeatureSetwithCluster(featureSetId, featureSetVersion, modelVersion, clusterDate, scoreDate, conn):
    print("====>clustered feature view")
    
    anomaly_features_names,anomaly_features_pos_weights,anomaly_features_neg_weights,ID = getClusteredFeatureWeights(featureSetId, featureSetVersion, conn)
    
    sql = f"""select {ID}, cluster_id, {anomaly_features_names}
        from {storage.featureStore_db_name}.v_modelDefinition_{featureSetId}_{featureSetVersion} md 
        JOIN {storage.dataScience_db_name}.v_lastest_object_cluster cr on ( 
            md.{ID} = cr.object_id 
            and cr.datascience_model_version = '{modelVersion}')
        where md.fc_agg_summary_date = {scoreDate}"""
        
    print(sql)

    X = pd.read_sql_query(sql,conn)
    return X, ID

def getModelMaxScore(featureSetId, featureSetVersion, modelVersion, conn):
    print("====>getModelMaxScore")
    
    sql = f"""select cluster_id, mf.feature,column_name,anomaly_pos_weight,anomaly_neg_weight,min_value, avg_value, max_value, std_value 
        from {storage.metadata_db_name}.v_model_feature mf 
        join {storage.dataScience_db_name}.cluster_explainability ce on (
            mf.column_name = ce.feature) 
        where is_anomaly = 1 and 
        model_id={featureSetId} and model_version = {featureSetVersion}
        and datascience_model_version = '{modelVersion}'
        order by cluster_id""" 
        
    print(sql)
    
    scoreModel = pd.read_sql_query(sql,conn)
    
    lastClusterId = 0
    clusterId = 0
    clusterMaxScore = 0.0
    clusterMinScore = 0.0
    clusterScore = {}
    clusterCount = 0
    
    for index, row in scoreModel.iterrows():
        print(f"procesing {row['feature']} for cluster {row['cluster_id']}")
        
        try:
            if row['anomaly_pos_weight'] > 0:
                featPosScoreMax = ((row['max_value'] - row['avg_value']) / row['std_value']) * row['anomaly_pos_weight']
                featPosScoreMin = 0
            else:
                featPosScoreMax = 0
                featPosScoreMin = ((row['max_value'] - row['avg_value']) / row['std_value']) * row['anomaly_pos_weight']
                
            if row['anomaly_neg_weight'] > 0:
                featNegScoreMax = abs((row['min_value'] - row['avg_value']) / row['std_value']) * row['anomaly_neg_weight']
                featNegScoreMin = 0
            else:
                featNegScoreMax = 0 
                featNegScoreMin = abs((row['min_value'] - row['avg_value']) / row['std_value']) * row['anomaly_neg_weight']

            featScoreMax = featPosScoreMax if featPosScoreMax > featNegScoreMax else featNegScoreMax
            featScoreMin = featNegScoreMin if featNegScoreMin < featPosScoreMin else featPosScoreMin

                
        except Exception as inst:
            featScoreMax = 0
            featScoreMin = 0
            featScoreMin = 0
        
        clusterId = int(row['cluster_id'])
        
        if clusterId == lastClusterId:
            #print('same cluster, increment scores')
            clusterMaxScore += (featScoreMax if featScoreMax > featScoreMin  else featScoreMin)
            clusterMinScore += (featScoreMin if featScoreMax > featScoreMin  else featScoreMax)
        else:
            #print('new cluster, start over and save score')
            clusterCount += 1
            clusterScore[f"cluster_{lastClusterId}_max"] = clusterMaxScore
            clusterScore[f"cluster_{lastClusterId}_min"] = clusterMinScore
            #print(clusterScore)
            clusterMaxScore = (featScoreMax if featScoreMax > featScoreMin  else featScoreMin)
            clusterMinScore = (featScoreMin if featScoreMax > featScoreMin  else featScoreMax)
            lastClusterId = int(clusterId)

    clusterCount += 1
    clusterScore[f"cluster_{lastClusterId}_max"] = clusterMaxScore
    clusterScore[f"cluster_{lastClusterId}_min"] = clusterMinScore
    clusterScore["cluster_count"] = clusterCount
    
    print(clusterScore)
            
    return scoreModel, clusterScore

def getStoredModel(model_version, conn):
    model_df = pd.read_sql(
        f"""select model from {storage.dataScience_db_name}.model_artifacts where model_version='{model_version}'""",
        conn)
    model = pickle.loads(model_df['model'][0])
    
    return model
