import pandas as pd
import numpy as np
from sklearn2pmml import sklearn2pmml
from sklearn.model_selection import train_test_split
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.ensemble import RandomForestClassifier
from teradataml.dataframe.copy_to import copy_to_sql
import pickle,shap,os,uuid,json,logging,time
from pypmml import Model
from sklearn import metrics
from datetime import datetime
from teradataml.context.context import *


def score(data_conf, model_conf, **kwargs):
    print ("dummy")


def get_engine(data_conf):
    host = data_conf["hostname"]
    username = "mk250108"
    password = ""
    logmech = "LDAP"
    eng = create_context(host=host, username=username, password=password, logmech=logmech)
    return eng


def create_connection(data_conf):
    eng = get_engine(data_conf)
    conn = eng.connect()
    return eng, conn


def get_test_data(data_conf, conn):
    test_data = pd.read_sql(f"""select * from {data_conf["ADS_db_name"]}.{data_conf["Evaluation_ADS_name"]}""",
                            conn)  # execute training data query
    # Separating labels and training data
    test_Y = test_data["fraud"]
    test_X = test_data.drop([data_conf["column_to_preserve"], "fraud"], axis=1)

    return test_X, test_Y

def get_prediction(x, threshold):
    if x >= threshold:
        return 1
    else:
        return 0

def evaluate(data_conf, model_conf, **kwargs):
    # Establish database connection
    eng, conn = create_connection(data_conf)
    model_version = kwargs.get("model_version")
    model_ID = kwargs.get("model_id")

    features_data, labels = get_test_data(data_conf, conn)

    model_df = pd.read_sql(
        f"""select model from {data_conf["datascience_db"]}.models_artifacts_ATO where model_version='{model_version}'""",
        conn)
    model = Model.load(model_df['model'][0])

    score_y = model.predict(features_data)
    print("Following are the model evaluation results. ")
    print(score_y)

    scores = {}

    # Calculate AUC Curve
    fpr, tpr, thresholds = metrics.roc_curve(labels, score_y["probability(1)"])
    auc = metrics.auc(fpr, tpr)
    GINI = (2 * auc) - 1
    
    print("step 1")

    scores['auc_'] = auc
    scores['gini_'] = GINI
    
    print("step 2")

    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    eval_df = pd.DataFrame(
        columns=["probability_threshold", "AUC", "GINI", "True_Positives", "True_Negatives", "False_Positives",
                 "False_Negatives", "Precision_NotFraud", "Recall_NotFraud", "Precision_Fraud", "Recall_Fraud",
                 "F1Score_Fraud", "F1Score_NotFraud"])
    
    print("step 3")

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
                                  "model_ID": model_ID,
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
    print("step 4")
    # dump results as json file evaluation.json to models/ folder
    with open("models/evaluation.json", "w+") as f:
        json.dump(scores, f)
        
    print("step 5")

    table_name = f"""Evaluation_{data_conf["Evaluation_ADS_name"]}_Mv_{model_version}_M_{model_ID}"""
    print(table_name)
    #table_name = "ZUZI"
    
    try:
        copy_to_sql(df=eval_df, table_name=table_name,schema_name=data_conf["datascience_db"],temporary=False,if_exists='replace')
        print("step 6")
    except Exception as e:
        print("%%%%%%%%%%%")
        print(e)
        print("**********")

    try:
        conn.execute(f"""UPDATE {data_conf["datascience_db"]}.global_explainability_metadata SET evaluation_table='{table_name}' WHERE model_version='{model_version}'""")
    except Exception as e:
        print("$$$$$$$$$$")
        print(e)
        print("##########")

    print("Evaluation complete...")
    
    
class ModelScorer(object):
    
    def predict(self, data):
        self.host = "tdprd2.td.teradata.com"
        self.username = "mk250108"
        self.password = ""
        self.logmech = "LDAP"
        self.eng = create_context(host=self.host, username=self.username, password=self.password, logmech=self.logmech)
        self.conn = self.eng.connect()
        self.model_df = pd.read_sql(
        f"""select skmodel from fincrime_cnp_dev.models_artifacts_ATO where model_version='1b4676b5-4d9a-4c5f-972c-a1f7431bc2f3'""",
        self.conn)
        self.model = pickle.loads(self.model_df['skmodel'][0])
        pred = self.model.predict_proba([data])[0][1]
        round_pred = round(pred,2)
        return round_pred
