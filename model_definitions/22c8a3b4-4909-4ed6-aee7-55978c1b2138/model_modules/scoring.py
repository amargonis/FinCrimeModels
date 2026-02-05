from datetime import datetime
import json
import pandas as pd
from teradataml.context.context import *
import numpy as np
from teradataml.dataframe.copy_to import copy_to_sql
from pypmml import Model
from sklearn import metrics
import logging


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


def get_test_data_query(data_conf):
    train_select = f"""select * from {data_conf["features_db"]}.{data_conf["evaluation_data"]} vslt"""
    return train_select


def get_score_data_query(data_conf):
    train_select = f"""select * from {data_conf["scoring_db"]}.{data_conf["scoring_data_object"]} vslt"""
    return train_select


def get_test_data(query, eng):
    test_data = pd.read_sql_query(query, eng)  # execute training data query
    # Separating labels and training data
    test_Y = test_data["fraud"]
    test_X = test_data.drop({"fraud", "cluster_id", "anomaly_score"}, axis=1).fillna(0)

    return test_X, test_Y


def get_prediction(x, threshold):
    if x >= threshold:
        return 1
    else:
        return 0


def evaluate(data_conf, model_conf, **kwargs):
    """Python evaluate method called by AOA framework
    Parameters:
    data_conf (dict): The dataset metadata
    model_conf (dict): The model configuration to use
    Returns:
    None:No return
    """

    """Python evaluate method called by AOA framework
    Parameters:
    data_conf (dict): The dataset metadata
    model_conf (dict): The model configuration to use
    Returns:
    None:No return
    """

    hyperparams = model_conf["hyperParameters"]
    model_version = kwargs.get("model_version")
    # model_version = "ModelVersion1"

    # Establish database connection
    eng, conn = create_connection(data_conf)

    train_select = get_test_data_query(data_conf)
    features_data, labels = get_test_data(train_select, eng)



    model = Model.load('models/model.pmml')
    score_y = model.predict(features_data)
    print("Following are the model evaluation results. ")
    print(score_y)

    scores = {}

    # Calculate AUC Curve
    fpr, tpr, thresholds = metrics.roc_curve(labels, score_y["probability(1)"])
    auc = metrics.auc(fpr, tpr)
    GINI = (2 * auc) - 1

    scores['auc_'] = auc
    scores['gini_'] = GINI

    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    eval_df = pd.DataFrame(
        columns=["probability_threshold", "AUC", "GINI", "True_Positives", "True_Negatives", "False_Positives",
                 "False_Negatives", "Precision_NotFraud", "Recall_NotFraud", "Precision_Fraud", "Recall_Fraud",
                 "F1Score_Fraud", "F1Score_NotFraud"])

    for probability_threshold in np.arange(1, 10, 1):  # Need to divide prob_threshold by 10 to resolve floating
        # point precision

        prediction = score_y["probability(1)"].apply(get_prediction,
                                                     args=({probability_threshold / 10}))

        # Get confusion matrix
        tn, fp, fn, tp = metrics.confusion_matrix(labels, prediction).ravel()

        classification_report = metrics.classification_report(labels, prediction, output_dict=True)

        scores['TruePositives_' + str(probability_threshold / 10)] = int(tp)
        scores['TrueNegatives_' + str(probability_threshold / 10)] = int(tn)
        scores['FalsePositives_' + str(probability_threshold / 10)] = int(fp)
        scores['FalseNegatives_' + str(probability_threshold / 10)] = int(fn)
        scores['Precision_NotFraud_' + str(probability_threshold / 10)] = classification_report["0"]["precision"]
        scores['Recall_NotFraud_' + str(probability_threshold / 10)] = classification_report["0"]["recall"]
        scores['Precision_Fraud_' + str(probability_threshold / 10)] = classification_report["1"]["precision"]
        scores['Recall_Fraud_' + str(probability_threshold / 10)] = classification_report["1"]["recall"]
        scores['f1-Score_NotFraud_' + str(probability_threshold / 10)] = classification_report["0"]["f1-score"]
        scores['f1-Score_Fraud_' + str(probability_threshold / 10)] = classification_report["1"]["f1-score"]

        eval_df = eval_df.append({"model_version": model_version,
                                  "timestamp": timestamp,
                                  "probability_threshold": probability_threshold / 10,
                                  "AUC": auc, "GINI": GINI,
                                  "True_Positives": tp, "True_Negatives": tn,
                                  "False_Positives": fp, "False_Negatives": fn,
                                  "Precision_NotFraud": classification_report["0"]["precision"],
                                  "Recall_NotFraud": classification_report["0"]["recall"],
                                  "Precision_Fraud": classification_report["1"]["precision"],
                                  "Recall_Fraud": classification_report["1"]["recall"],
                                  "F1Score_Fraud": classification_report["0"]["f1-score"],
                                  "F1Score_NotFraud": classification_report["1"]["f1-score"]
                                  }, ignore_index=True)

    # dump results as json file evaluation.json to models/ folder
    with open("models/evaluation.json", "w+") as f:
        json.dump(scores, f)

    table_name = data_conf['evaluation_tbl']

    copy_to_sql(df=eval_df, table_name=table_name,
                schema_name=data_conf["features_db"],
                temporary=False,
                if_exists='append')

    print("Evaluation complete...")


def score(data_conf, model_conf, db=None, deploy=True, **kwargs):
    """Python function to perform scoring
    Parameters:
    :param data_conf (dict): The dataset metadata
    :param model_conf (dict): The model configuration to use
    :param db (object): db object with connection and engine info
    :param deploy (bool): scoring or evaluation
    Returns:
    None:No return
    """

    logging.basicConfig(level=logging.INFO)
    logging.info("Scoring Started")

    print("Scoring started..")

    conn, eng = create_connection(data_conf)
    test_select = get_score_data_query(data_conf)

    # model_version = kwargs.get("model_version")

    model_version = "model_version1"

    print("Model Version: ")
    print(model_version)

    # Get model ID
    query = f"""select model_id from {data_conf['scoring_db']}.{data_conf['scoring_model_artifacts_table']} WHERE 
model_version='{model_version}' """

    model_id = pd.read_sql_query(query, conn)["model_id"].to_string(index=False).replace(" ", "")

    model_id = "model_id1"

    # Query to execute.
    ivsm_scoring_query = f"""
      select * from adldemo_ivsm.IVSM_score2(
          on ({test_select})
          on (select model_id, model from {data_conf['scoring_db']}.{data_conf['scoring_model_artifacts_table']} WHERE model_version='{model_version}') DIMENSION 
          using
              ModelID('{model_id}')
              ColumnsToPreserve('{data_conf["column_to_preserve"]}') 
              ModelType('PMML')
              ModelSpecificSettings('PMML_OUTPUT_TYPE=ALL')
      ) sc;
      """

    predict_table = data_conf['predict_tbl']

    print("Start scoring...")

    scoring_result = pd.read_sql_query(ivsm_scoring_query, conn)
    prob_1 = scoring_result["score_result"].apply(lambda x: json.loads(x)["probability(1)"]).astype(float)

    scoring_result.insert(2, "probability_1", prob_1)
    scoring_result.insert(3, "probability_0", 1 - scoring_result["probability_1"])
    prediction = scoring_result["probability_1"].apply(get_prediction,
                                                       args=({model_conf["hyperParameters"]["probability_threshold"]}))
    scoring_result.insert(4, "prediction", prediction)
    scoring_result.insert(5, "model_version", model_version)

    copy_to_sql(df=scoring_result, table_name=predict_table, schema_name=data_conf["scoring_db"], temporary=False,
                if_exists='replace')

    print("End scoring")
    logging.info("End scoring")


# Uncomment this code if you want to deploy your model as a Web Service (Real-time / Interactive usage)
# class ModelScorer(object):
#    def __init__(self, config=None):
#        self.model = joblib.load('models/iris_knn.joblib')
#
#    def predict(self, data):
#        return self.model.predict([data])
#

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
#             "column_to_preserve": "partyid",
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
#         score(data_conf, model_conf)
#
#     except Exception as e:
#         print(e)
