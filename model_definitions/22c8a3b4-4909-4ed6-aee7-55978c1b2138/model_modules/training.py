import os
import pandas as pd
import teradataml as tdml
import numpy as np
import json

from sklearn2pmml import sklearn2pmml
from teradataml.context.context import *
from teradataml.dataframe.copy_to import copy_to_sql

# import lightgbm as lgb
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split
from sklearn2pmml.decoration import ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.ensemble import RandomForestClassifier
# from pypmml import Model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


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


def get_training_data_query(data_conf):
    train_select = f"""select * from {data_conf["features_db"]}.{data_conf["model_features"]} vslt"""
    return train_select


def get_training_data(query, eng):
    training_data = pd.read_sql_query(query, eng)  # execute training data query
    # Separating labels and training data
    training_Y = training_data["fraud"]
    training_X = training_data.drop({"fraud", "cluster_id", "anomaly_score"}, axis=1).fillna(0)

    return training_X, training_Y


def delete_record_if_exists(db_name, model_table, model_version, conn):
    ret = pd.read_sql_query(f"select * from {db_name}.{model_table} WHERE model_version='{model_version}'", conn)

    if len(ret) > 0:
        conn.execute("delete from " + db_name + "." + model_table + " WHERE model_version='" + model_version + "'")
        print("model deleted")


def if_not_exist_create_table(db_name, tbl_name, conn):
    ret = pd.read_sql(
        f"select distinct TableName FROM DBC.TablesV WHERE DatabaseName= '{db_name}' and TableName='{tbl_name}'", conn)

    # if not len(ret) > 0:
    #     conn.execute(f"""CREATE MULTISET TABLE {db_name}.{tbl_name} (model_id VARCHAR(100), model
    # BLOB, model_version VARCHAR(100), deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")

    if not len(ret) > 0:
        conn.execute(f"""CREATE MULTISET TABLE {db_name}.{tbl_name}, FALLBACK , NO BEFORE JOURNAL,
                        NO AFTER JOURNAL, CHECKSUM = DEFAULT, DEFAULT MERGEBLOCKRATIO, MAP = TD_MAP1
                        (
                        model_version VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC, model_id VARCHAR(255) CHARACTER 
                        SET LATIN CASESPECIFIC, deployed_at 
                        TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
                        model BLOB(2097088000) )
                        UNIQUE PRIMARY INDEX ( model_version );""")


def loadModel(db_name, model_table, conn, model_id, model_version):
    if_not_exist_create_table(db_name, model_table, conn)
    delete_record_if_exists(db_name, model_table, model_version, conn)

    model_bytes = open("models/model.pmml", "rb").read()

    conn.execute(f"insert into " + db_name + "." + model_table + "(model_id, model, model_version) values(?,?,?)",
                 model_id, model_bytes, model_version)

    print("model dump done.")


def train(data_conf, model_conf, **kwargs):
    """Python train method called by AOA framework

    Parameters:
    data_conf (dict): The dataset metadata
    model_conf (dict): The model configuration to use

    Returns:
    None:No return

    """

    hyperparams = model_conf["hyperParameters"]
    # model_version = kwargs.get("model_version")
    # model_id = kwargs.get("model_id")

    model_version = "model_version1"
    model_id = "model_id1"

    # Establish database connection
    eng, conn = create_connection(data_conf)

    train_select = get_training_data_query(data_conf)
    features, labels = get_training_data(train_select, eng)

    # load data & engineer

    print("Starting training...")

    classifier = PMMLPipeline([('classifier', RandomForestClassifier(n_estimators=hyperparams["n_estimators"],
                                                                     max_depth=hyperparams["max_depth"],
                                                                     min_samples_split=hyperparams["min_samples_split"],
                                                                     min_samples_leaf=hyperparams["min_samples_leaf"],
                                                                     max_features=hyperparams["max_features"],
                                                                     min_impurity_decrease=hyperparams[
                                                                         "min_impurity_decrease"],
                                                                     bootstrap=hyperparams["bootstrap"],
                                                                     oob_score=hyperparams["oob_score"],
                                                                     verbose=hyperparams["verbose"],
                                                                     warm_start=hyperparams["warm_start"],
                                                                     ccp_alpha=hyperparams["ccp_alpha"],
                                                                     ))])

    # fit model to training data
    classifier.fit(features, labels)
    print("Finished training")

    # export model artefacts to models/ folder
    if not os.path.exists('models'):
        os.makedirs('models')
    sklearn2pmml(classifier, "models/model.pmml")
    print("Saved trained model")

    # Uplaod model to Vantage
    loadModel(data_conf['scoring_db'], data_conf['scoring_model_artifacts_table'], conn, model_id, model_version)  # Load model to db


# if __name__ == "__main__":
#     try:
#         data_conf = {
#             "hostname": "tdprd2.td.teradata.com",
#             "src_db": "fincrime_src_dev",
#             "features_db": "fincrime_bustout_dev",
#             "db_name": "fincrime_aml_dev",
#             "mdl_id": "aml_superv_00",
#             "party_info": "party",
#             "dataset": "transaction_agg",
#             "train_test_split": "train_test_instances",
#             "model_features": "v_supervised_learning_train",
#             "feature_metadata": "feature_metadata",
#             "evaluation_data": "v_supervised_learning_test",
#             "scoring_db": "fincrime_bustout_dev",
#             "scoring_model_artifacts_table": "mode_artifacts_table",
#             "scoring_data_object": "v_supervised_learning_test",
#             "column_to_preserve": "model_version",
#             "predict_tbl": "fincrime_bustout_ivsm_scoring",
#             "predict_tbl_for_roc": "fc_aml_predictions_for_roc",
#             "evaluation_tbl": "fincrime_bustout_dev_evaluation"
#         }
#
#         with open("../config.json") as json_file:
#             model_conf = json.load(json_file)
#
#         with open("cred.json") as json_file:
#             creds = json.load(json_file)
#
#         os.environ["TD_USERNAME"] = creds["TD_Username"]
#         os.environ["TD_PASSWORD"] = creds["TD_Password"]
#
#         train(data_conf, model_conf)
#
#     except Exception as e:
#         print(e)
