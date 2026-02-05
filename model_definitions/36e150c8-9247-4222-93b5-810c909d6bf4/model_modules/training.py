import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from teradataml.dataframe.copy_to import copy_to_sql
import pickle, os
from teradataml.context.context import *
from sklearn.linear_model import LogisticRegression



def get_engine(data_conf):
    host = data_conf["hostname"]
    username = os.environ["TD_USERNAME"]
    password = os.environ["TD_PASSWORD"]
    logmech = os.getenv("TD_LOGMECH", "TDNEGO")
    eng = create_context(host=host, username=username, password=password, logmech=logmech)
    return eng


def create_connection(data_conf):
    eng = get_engine(data_conf)
    conn = eng.connect()
    return eng, conn


def meta_data(data_conf, conn):
    meta_data = pd.read_sql(
        f"""select model_version,model_id,feature,is_cluster,ml_type from {data_conf["metadata_db_name"]}.{data_conf["metadata_dataset_name"]} where model_id={data_conf["dataset_ID"]} and model_version = {data_conf["dataset_version"]}""",
        conn)
    cluster_features_names = ",".join(
        meta_data[(meta_data['ml_type'] == 2) & (meta_data['is_cluster'] == 1)]['feature'].tolist())
    ID = meta_data[meta_data['ml_type'] == 1]['feature'].values[0]
    dataset_version = meta_data['model_version'].values[0]
    dataset_ID = meta_data['model_id'].values[0]
    return cluster_features_names, ID, dataset_version, dataset_ID


def delete_record_if_exists(db_name, model_version, conn):
    conn.execute(f"""delete from {db_name}.models_artifacts_AML WHERE model_version='{model_version}'""")


def if_not_exist_create_table(db_name, conn):
    ret = pd.read_sql(
        f"select distinct TableName FROM DBC.TablesV WHERE DatabaseName= '{db_name}' and TableName='models_artifacts_AML'",
        conn)

    if not len(ret) > 0:
        conn.execute(f"""CREATE MULTISET TABLE {db_name}.models_artifacts_AML, FALLBACK , NO BEFORE JOURNAL,
                        NO AFTER JOURNAL, CHECKSUM = DEFAULT, DEFAULT MERGEBLOCKRATIO, MAP = TD_MAP1
                        (
                        model_version VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC,
                        model_id VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC,
                        deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        model BLOB(2097088000),
                        dataset_version VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC,
                        dataset_id VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC
                        )
                        UNIQUE PRIMARY INDEX ( model_version );""")


def loadModel(data_conf, clustered_model, dataset_version, dataset_ID, model_version, model_ID, conn):
    if_not_exist_create_table(data_conf["datascience_db"], conn)
    delete_record_if_exists(data_conf["datascience_db"], str(model_version), conn)

    clustered_model_bytes = pickle.dumps(clustered_model)
    conn.execute(
        f"""insert into {data_conf["datascience_db"]}.models_artifacts_AML(model_version, model_id, model, 
dataset_version, dataset_ID) values(?,?,?,?,?)""",
        str(model_version), str(model_ID), clustered_model_bytes, str(dataset_version), str(dataset_ID))

    print("model dump done.")


def cluster_explainability(data, no_of_clusters, centroids, model_version, model_ID, dataset_version, dataset_ID, data_conf):
    models = {}
    g_exp = pd.DataFrame(
        columns=["feature", "importance", "cluster_ID", "centroid"])

    for i in range(no_of_clusters):
        positive_cluster = data[data["cluster_ID"] == i]
        negative_cluster = data[data["cluster_ID"] != i]
        positive_cluster["cluster_ID"] = positive_cluster["cluster_ID"].apply(lambda x: 1)
        negative_cluster["cluster_ID"] = negative_cluster["cluster_ID"].apply(lambda x: 0)

        frames = [positive_cluster, negative_cluster]

        clustered_data = pd.concat(frames)

        X = clustered_data.drop({"cluster_ID", "fc_agg_summary_date"}, axis=1)
        y = clustered_data["cluster_ID"]

        model = RandomForestClassifier()
        model.fit(X, y)

        models[i] = model

        feats = {}  # a dict to hold feature_name: feature_importance

        for feature, importance, centroid in zip(X.columns, model.feature_importances_, centroids.iloc[i]):
            # feats[feature] = importance
            # feats[feature] = i
            # feats[feature] = centroids.iloc[i].values
            # #feats.update(centroids.iloc[i])

            g_exp = g_exp.append({"feature": feature,
                          "importance": importance,
                          "cluster_ID": i,
                          "centroid": centroid
                          }, ignore_index=True)

    copy_to_sql(g_exp,
                table_name=f"""Clusters_exp_Mv_{model_version}_Mid_{model_ID}_Dv_{dataset_version}_Did_{dataset_ID}""",
                if_exists="replace", schema_name=data_conf["datascience_db"])


def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]
    eng, conn = create_connection(data_conf)

    model_version = kwargs.get("model_version")
    model_ID = kwargs.get("model_id")

    cluster_features_names, ID, dataset_version, dataset_ID = meta_data(data_conf, conn)
    X = pd.read_sql(
        f"""select {ID},{data_conf['Date_col_name']},{cluster_features_names} from {data_conf["ADS_db_name"]}.{data_conf["ADS_name"]}""",
        conn)

    IDs_date = X[[ID, data_conf["Date_col_name"]]]
    X.drop([ID, data_conf["Date_col_name"]], inplace=True, axis=1)
    X.fillna(0, inplace=True)
    cluster_features_scaled = StandardScaler().fit_transform(X)

    km_silhouette = []
    maximum = 0
    clusters = 1
    for i in range(2, data_conf["n_clusters"]):
        km = KMeans(n_clusters=i, init=hyperparams["init"], max_iter=hyperparams["max_iter"],
                    n_init=hyperparams["n_init"]).fit(cluster_features_scaled)
        preds = km.predict(cluster_features_scaled)
        silhouette = silhouette_score(cluster_features_scaled, preds)
        if silhouette > maximum:
            maximum = silhouette
            clusters = i
        km_silhouette.append(silhouette)
        print("Silhouette score for number of cluster(s) {}: {}".format(i, silhouette))

    km = KMeans(n_clusters=clusters, init=hyperparams["init"], max_iter=hyperparams["max_iter"],
                             n_init=hyperparams["n_init"])

    clustered_model = km.fit(cluster_features_scaled)
    centroids = pd.DataFrame(km.cluster_centers_, columns=list(map(lambda x: str(x) + "_centroid", cluster_features_names.split(","))))
    loadModel(data_conf, clustered_model, dataset_version, dataset_ID, model_version, model_ID, conn)
    cluster_all_data = pd.concat([IDs_date, X], axis=1)
    preds = clustered_model.predict(cluster_features_scaled)
    cluster_all_data['cluster_ID'] = preds

    copy_to_sql(cluster_all_data,
                table_name=f"""Clustering_Results_Mv_{model_version}_Mid_{model_ID}_Dv_{dataset_version}_Did_{dataset_ID}""",
                if_exists="replace",
                schema_name=data_conf["datascience_db"])
    
    cluster_all_data.drop(ID, inplace=True, axis=1)
    cluster_explainability(cluster_all_data, clusters, centroids, str(model_version), str(model_ID), str(dataset_version), str(dataset_ID), data_conf)


if __name__ == "__main__":
    try:
        data_conf = {
            "hostname": "tdprd2.td.teradata.com",
            "ADS_db_name": "fincrime_aml_dev",
            "ADS_name": "AML_ADS_MS_1",
            "Date_col_name": "fc_agg_summary_date",
            "metadata_db_name": "fincrime_aml_dev",
            "metadata_dataset_name": "model_feature3",
            "dataset_version": "1",
            "dataset_ID": "100001",
            "datascience_db": "fincrime_aml_dev",
            "alert_db": "fincrime_aml_dev",
            "n_clusters": 5
            }

        # "hyperParameters": {
        #     "init": "k-means++",
        #     "max_iter": 300,
        #     "n_init": 10
        # }


        with open("../config.json") as json_file:
            model_conf = json.load(json_file)

        with open("cred.json") as json_file:
            creds = json.load(json_file)

        os.environ["TD_USERNAME"] = creds["TD_Username"]
        os.environ["TD_PASSWORD"] = creds["TD_Password"]

        train(data_conf, model_conf)

    except Exception as e:
        print(e)
