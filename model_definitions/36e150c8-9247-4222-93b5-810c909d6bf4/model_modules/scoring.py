import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle, os, json
from teradataml.dataframe.copy_to import copy_to_sql
from teradataml.context.context import *
from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext
)

def get_engine(data_conf):
    host = "tdprd2.td.teradata.com"
    username = 'mk250108'
    password = ""
    logmech = "LDAP"
    eng = create_context(host=host, username=username, password=password, logmech=logmech)
    return eng


def create_connection(data_conf):
    eng = get_engine(data_conf)
    conn = eng.connect()
    return eng, conn


def meta_data(data_conf, conn):
    meta_data = pd.read_sql(
        f"""select model_version,model_id,feature,is_cluster,is_anomaly,anomaly_pos_weight,anomaly_neg_weight,ml_type from {data_conf["metadata_db_name"]}.{data_conf["metadata_dataset_name"]} where model_id={data_conf["dataset_ID"]} and model_version = {data_conf["dataset_version"]}""",
        conn)
    cluster_features_names = meta_data[(meta_data['ml_type'] == 2) & (meta_data['is_cluster'] == 1)]['feature'].tolist()
    anomaly_features_names = meta_data[(meta_data['ml_type'] == 2) & (meta_data['is_anomaly'] == 1)]['feature'].tolist()
    anomaly_features_pos_weights = meta_data[(meta_data['ml_type'] == 2) & (meta_data['is_anomaly'] == 1)][
        'anomaly_pos_weight']
    anomaly_features_neg_weights = meta_data[(meta_data['ml_type'] == 2) & (meta_data['is_anomaly'] == 1)][
        'anomaly_neg_weight']
    ID = meta_data[meta_data['ml_type'] == 1]['feature'].values[0]
    dataset_version = meta_data['model_version'].values[0]
    dataset_ID = meta_data['model_id'].values[0]
    return cluster_features_names, anomaly_features_names, anomaly_features_pos_weights, anomaly_features_neg_weights, ID, dataset_version, dataset_ID


def evaluate(data_conf, model_conf, **kwargs):
    scores = {}
    scores['auc_'] = "dummy"
    with open("models/evaluation.json", "w+") as f:
        json.dump(scores, f)
    # dump results as json file evaluation.json to models/ folder
    print("Evaluation complete...")


def score((context: ModelContext , db=None, deploy=True, **kwargs):
    print("Scoring started..")
    
    aoa_create_context()
    data_conf = context.dataset_info
    hyperparams = context.hyperParameters

    eng, conn = create_connection(data_conf)

    model_version = kwargs.get("model_version")
    query = f"""select model_id from {data_conf["datascience_db"]}.models_artifacts_AML WHERE 
    model_version='{model_version}'"""
    model_ID = pd.read_sql_query(query, conn)["model_id"].to_string(index=False).replace(" ", "")

    cluster_features_names, anomaly_features_names, anomaly_features_pos_weights, anomaly_features_neg_weights, ID, dataset_version, dataset_ID = meta_data(
        data_conf, conn)

    anomaly_features_pos_weights = anomaly_features_pos_weights.replace(0, 1)
    anomaly_features_neg_weights = anomaly_features_neg_weights.replace(0, 1)
    anomaly_features_pos_weights = anomaly_features_pos_weights / anomaly_features_pos_weights.sum()
    anomaly_features_neg_weights = anomaly_features_neg_weights / anomaly_features_neg_weights.sum()
    lst_anomaly_features_pos_weights = anomaly_features_pos_weights.to_list()
    lst_anomaly_features_neg_weights = anomaly_features_neg_weights.to_list()

    X = pd.read_sql(f"""select * from {data_conf["Score_ADS_db_name"]}.{data_conf["Score_ADS_name"]}""", conn)

    IDs_date = X[[ID, data_conf["Date_col_name"]]]
    X.drop([ID, data_conf["Date_col_name"]], inplace=True, axis=1)
    X.fillna(0, inplace=True)

    cluster_features = X[cluster_features_names]
    anomaly_features = X[anomaly_features_names]

    cluster_features_scaled = StandardScaler().fit_transform(cluster_features)

    model_df = pd.read_sql(
        f"""select model from {data_conf["datascience_db"]}.models_artifacts_AML where model_version='{model_version}'""",
        conn)
    model = pickle.loads(model_df['model'][0])

    preds = model.predict(cluster_features_scaled)
    anomaly_features['cluster_ID'] = preds

    ranks_anomaly = anomaly_features.groupby('cluster_ID').rank(pct=True)
    ranks_anomaly = 0.5 - ranks_anomaly
    ranks_anomaly[ranks_anomaly < 0] *= lst_anomaly_features_pos_weights
    ranks_anomaly[ranks_anomaly > 0] *= lst_anomaly_features_neg_weights

    weighted_distances_anomaly = 2.0 * (ranks_anomaly).abs()

    weighted_distances_anomaly.columns = [str(col) + '_weighted_dist' for col in ranks_anomaly.columns]

    weighted_distances_anomaly['accumulated_score'] = weighted_distances_anomaly.mean(axis=1) * 100
    anom_score_min = weighted_distances_anomaly['accumulated_score'].min()
    anom_score_range = weighted_distances_anomaly['accumulated_score'].max() - anom_score_min
    weighted_distances_anomaly['anomaly_score'] = (weighted_distances_anomaly[
                                                       'accumulated_score'] - anom_score_min) / anom_score_range

    anomaly_features_all_data = pd.concat([anomaly_features, weighted_distances_anomaly], axis=1)
    anomaly_all_data = pd.concat([IDs_date, anomaly_features_all_data], axis=1)

    last_col = anomaly_all_data.pop('cluster_ID')
    anomaly_all_data['cluster_ID'] = last_col

    copy_to_sql(anomaly_all_data,
                table_name=f"""Anomaly_Results_Mv_{model_version}_Mid_{model_ID}_Dv_{dataset_version}_Did_{dataset_ID}""",
                if_exists="replace",
                schema_name=data_conf["datascience_db"])

