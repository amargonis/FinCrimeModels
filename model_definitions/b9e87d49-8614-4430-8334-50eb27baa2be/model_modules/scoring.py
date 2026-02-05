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


def get_engine(data_conf):
    host = data_conf["hostname"]
    username = os.getenv("TD_USERNAME")
    password = os.getenv("TD_PASSWORD")
    logmech = os.getenv("TD_LOGMECH", "TDNEGO")
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


def explainability(scoring_result, data_conf, model_version, model_ID, conn, timestamp):
    score_ADS = pd.read_sql(f"""select * from {data_conf["ADS_db_name"]}.{data_conf["Score_ADS_name"]}""", conn)
    as_of_date = str(score_ADS[data_conf["Date_col_name"]].values[0])
    score_ADS = score_ADS.drop(data_conf["Date_col_name"], axis=1)
    score_ADS = score_ADS.astype(float)
    model_df = pd.read_sql(
        f"""select skmodel from {data_conf["ADS_db_name"]}.models_artifacts_BO where model_version='{model_version}'""",
        conn)
    model = pickle.loads(model_df['skmodel'][0])
    X_test = score_ADS.drop(data_conf["column_to_preserve"], axis=1)

    score_ADS['prob'] = model.predict_proba(X_test)[:, 1]
    local_ADS = score_ADS[score_ADS['prob'] >= data_conf["local_explainabilty_threshold"]]
    local_ADS = local_ADS.reset_index(drop=True)
    local_ADS = local_ADS.drop(["prob"], axis=1)
    X_local = local_ADS.drop([data_conf["column_to_preserve"]], axis=1)
    explainer = shap.Explainer(model.predict, X_local)

    shap_values = explainer(X_local)
    shap_v = pd.DataFrame(shap_values.values)
    shap_v.columns = X_local.columns
    shap_v.columns = [str(col) + '_exp' for col in shap_v.columns]
    #shap_abs = np.abs(shap_v)
    l = pd.concat([local_ADS[data_conf["column_to_preserve"]], shap_v], axis=1)
    # scoring_result['acct_no'] = scoring_result['acct_no'].apply(np.int64)

    # merge local exp with scoring.
    local_exp = l.merge(scoring_result, on="acct_no")
    local_exp.acct_no = local_exp.acct_no.astype(int)

    copy_to_sql(local_exp, table_name=f"""Local_Interpretability_Mv_{model_version}_M_{model_ID}_{timestamp}""",
                if_exists="replace", schema_name=data_conf["datascience_db"])
    
    alert_ads = local_exp[[data_conf["column_to_preserve"],"Score"]]
    
    return as_of_date, alert_ads
    
    
def process_local_explainability_metadata(model_version, model_ID, data_conf, as_of_date ,timestamp):
    # local explainabilit metadata
    local_exp_metadata = pd.DataFrame(
        columns=["model_version", "model_id", "database_name",
                 "local_explainability_table", "scoring_table", "as_of_date", "job_trigger_timestamp"])

    local_exp_metadata = local_exp_metadata.append({"model_version": model_version,
                                                    "model_id": model_ID,
                                                    "database_name": data_conf["datascience_db"],
                                                    "local_explainability_table": f"""Local_Interpretability_Mv_{model_version}_M_{model_ID}_{timestamp}""",
                                                    "scoring_table":data_conf["Score_ADS_name"],
                                                    "as_of_date": as_of_date,
                                                    "job_trigger_timestamp": timestamp
                                                    }, ignore_index=True)

    copy_to_sql(df=local_exp_metadata, table_name="local_explainability_metadata",
                schema_name=data_conf["datascience_db"],
                temporary=False,
                if_exists='append')
    
    print("Explainability complete...")
    

def create_tables(data_conf):

    supp_alerts = pd.DataFrame(columns=['alert_id','parent_alert_id','object_id','object_type','crime_type','suppression_reason', 'mute_start_date',
                                            'mute_end_date', 'recency_value',
                                            'similarity_value'])
    alerts = pd.DataFrame(columns=['alert_id', 'alert_status', 'crime_type', 'model_id', 'model_version', 'date_time',
                                   'as_of_date', 'resolution_date', 'object_id', 'object_type',
                                   'probability', 'decision_threshold', 'priority', 'last_updated_by'])

    copy_to_sql(supp_alerts, table_name="alerts_suppressed",
                    if_exists="append", schema_name=data_conf["alert_db"])
    
    copy_to_sql(alerts, table_name="alerts",
                    if_exists="append", schema_name=data_conf["alert_db"])
    
    
def convert_date(str_date_time):
    return datetime.strptime(str_date_time, '%d/%m/%Y_%H:%M:%S')

def reconvert_date(dateObj):
    return dateObj.strftime("%d/%m/%Y_%H:%M:%S")

def suppress_custom_alert(data_conf, alert_ads, crime_type, model_version , model_ID, as_of_date, conn):
    ID = data_conf["column_to_preserve"]
    str_IDs = ','.join(str(v) for v in alert_ads[ID].to_list())
    
    alerts_in_db = pd.read_sql(
        f"""select object_id,alert_id,mute_start_date,mute_end_date from {data_conf["alert_db"]}.alerts_suppressed WHERE mute_start_date is not null and mute_end_date is not null and crime_type = '{crime_type}' and object_id in ({str_IDs})""",
        conn)
    alerts_in_db.object_id = alerts_in_db.object_id.astype(int)
    alerts_in_db.mute_start_date = alerts_in_db.mute_start_date.apply(convert_date)
    alerts_in_db.mute_end_date = alerts_in_db.mute_end_date.apply(convert_date)
        
    alerts_in_db = alerts_in_db[(alerts_in_db["mute_start_date"] <= datetime.now()) & (alerts_in_db["mute_end_date"] >= datetime.now())]
    alerts_in_db.mute_start_date = alerts_in_db.mute_start_date.apply(reconvert_date)
    alerts_in_db.mute_end_date = alerts_in_db.mute_end_date.apply(reconvert_date)
    
    df_all = alert_ads.merge(alerts_in_db.drop_duplicates(), left_on=ID, right_on = 'object_id', how='left', indicator=True)
    present_acct_number_rows = df_all[df_all['_merge'] == 'both']
    del present_acct_number_rows['_merge']
    absent_acct_number_rows = df_all[df_all['_merge'] == 'left_only']
    del absent_acct_number_rows['_merge']
    
    if not present_acct_number_rows.empty:
        alerts_df = pd.DataFrame(columns=['alert_id', 'alert_status', 'crime_type','model_id', 'model_version', 'date_time',
                                       'as_of_date', 'resolution_date', 'object_id', 'object_type',
                                       'probability', 'decision_threshold', 'priority', 'last_updated_by'])

        alerts_df['object_id'] = present_acct_number_rows[ID]
        alerts_df["object_type"] = ID
        alerts_df["crime_type"] = crime_type
        alerts_df['date_time'] = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
        alerts_df['as_of_date'] = as_of_date
        alerts_df["alert_id"] = alerts_df["alert_id"].apply(lambda _: uuid.uuid4().hex)
        alerts_df["model_id"] = model_ID
        alerts_df["model_version"] = model_version
        alerts_df["alert_status"] = "suppressed"
        alerts_df["resolution_date"] = None
        alerts_df["probability"] = present_acct_number_rows["Score"]
        alerts_df["decision_threshold"] = 0.5
        alerts_df["priority"] = None
        alerts_df["last_updated_by"] = "system"

        copy_to_sql(alerts_df, table_name="alerts",
                        if_exists="append", schema_name=data_conf["alert_db"])
                        

        supp_alerts_df = pd.DataFrame(columns=['alert_id','parent_alert_id','object_id','object_type','crime_type','suppression_reason', 'mute_start_date',
                                                'mute_end_date', 'recency_value',
                                                'similarity_value'])

        supp_alerts_df["alert_id"] = alerts_df["alert_id"]
        supp_alerts_df["parent_alert_id"] = present_acct_number_rows["alert_id"]
        supp_alerts_df['object_id'] = alerts_df['object_id']
        supp_alerts_df["object_type"] = ID
        supp_alerts_df["crime_type"] = crime_type
        supp_alerts_df["mute_start_date"] = present_acct_number_rows["mute_start_date"]
        supp_alerts_df["mute_end_date"] = present_acct_number_rows["mute_end_date"]
        supp_alerts_df["suppression_reason"] = "end_user_custom"
        supp_alerts_df["recency_value"] = None
        supp_alerts_df["similarity_value"] = None

        copy_to_sql(supp_alerts_df, table_name="alerts_suppressed",
                        if_exists="append", schema_name=data_conf["alert_db"])
                        
                    
    carry_over_custom = absent_acct_number_rows.drop(["alert_id","object_id","mute_start_date","mute_end_date"], axis=1)            
    return carry_over_custom

def suppress_open_alert(data_conf, carry_over_custom, crime_type, model_version , model_ID, as_of_date, conn):
    ID = data_conf["column_to_preserve"]
    str_IDs = ','.join(str(v) for v in carry_over_custom[ID].to_list())
    lst_IDs = carry_over_custom[ID].to_list()
    alerts_in_db = pd.read_sql(
        f"""select object_id,alert_id from {data_conf["alert_db"]}.alerts WHERE object_id in ({str_IDs}) and crime_type = '{crime_type}' and alert_status = 'open'""",
        conn)
        
    alerts_in_db.object_id = alerts_in_db.object_id.astype(int)
    
    df_all = carry_over_custom.merge(alerts_in_db.drop_duplicates(), left_on=ID, right_on = 'object_id', how='left', indicator=True)
    present_acct_number_rows = df_all[df_all['_merge'] == 'both']
    absent_acct_number_rows = df_all[df_all['_merge'] == 'left_only']
    del absent_acct_number_rows['_merge']
    
    alerts_df = pd.DataFrame(columns=['alert_id', 'alert_status', 'crime_type', 'model_id', 'model_version', 'date_time',
                                   'as_of_date', 'resolution_date', 'object_id', 'object_type',
                                   'probability', 'decision_threshold', 'priority', 'last_updated_by'])
    
    alerts_df['object_id'] = present_acct_number_rows[ID]
    alerts_df["object_type"] = ID
    alerts_df["crime_type"] = crime_type
    alerts_df['date_time'] = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
    alerts_df['as_of_date'] = as_of_date
    alerts_df["alert_id"] = alerts_df["alert_id"].apply(lambda _: uuid.uuid4().hex)
    alerts_df["model_id"] = model_ID
    alerts_df["model_version"] = model_version
    alerts_df["alert_status"] = "suppressed"
    alerts_df["resolution_date"] = None
    alerts_df["probability"] = present_acct_number_rows["Score"]
    alerts_df["decision_threshold"] = data_conf["alert_threshold"]
    alerts_df["priority"] = None
    alerts_df["last_updated_by"] = "system"

    copy_to_sql(alerts_df, table_name="alerts",
                    if_exists="append", schema_name=data_conf["alert_db"])
                    
    supp_alerts_df = pd.DataFrame(columns=['alert_id','parent_alert_id','object_id','object_type','crime_type','suppression_reason', 'mute_start_date',
                                            'mute_end_date', 'recency_value',
                                            'similarity_value'])
    
    supp_alerts_df["alert_id"] = alerts_df["alert_id"]
    supp_alerts_df["parent_alert_id"] = present_acct_number_rows["alert_id"]
    supp_alerts_df['object_id'] = alerts_df['object_id']
    supp_alerts_df["object_type"] = ID
    supp_alerts_df["crime_type"] = crime_type
    supp_alerts_df["mute_start_date"] = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
    supp_alerts_df["mute_end_date"] = None
    supp_alerts_df["suppression_reason"] = "already open"
    supp_alerts_df["recency_value"] = None
    supp_alerts_df["similarity_value"] = None
    
    copy_to_sql(supp_alerts_df, table_name="alerts_suppressed",
                    if_exists="append", schema_name=data_conf["alert_db"])
                    
    carry_over_open = absent_acct_number_rows.drop(["alert_id","object_id"], axis=1)            
    return carry_over_open


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

def open_alert(data_conf, carry_over_closed, crime_type, model_version , model_ID, as_of_date, conn):
    ID = data_conf["column_to_preserve"]
    
    alerts_df = pd.DataFrame(columns=['alert_id','alert_status','crime_type','model_id', 'model_version', 'date_time',
                               'as_of_date', 'resolution_date', 'object_id', 'object_type',
                               'probability', 'decision_threshold', 'priority', 'last_updated_by'])

    alerts_df['object_id'] = carry_over_closed[ID]
    alerts_df["object_type"] = ID
    alerts_df["crime_type"] = crime_type
    alerts_df['date_time'] = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
    alerts_df['as_of_date'] = as_of_date
    alerts_df["alert_id"] = alerts_df["alert_id"].apply(lambda _: uuid.uuid4().hex)
    alerts_df["model_id"] = model_ID
    alerts_df["model_version"] = model_version
    alerts_df["alert_status"] = "open"
    alerts_df["resolution_date"] = None
    alerts_df["probability"] = carry_over_closed["Score"]
    alerts_df["decision_threshold"] = data_conf["alert_threshold"]
    alerts_df["priority"] = carry_over_closed["Score"].apply(get_priority_level)
    alerts_df["last_updated_by"] = "system"

    copy_to_sql(alerts_df, table_name="alerts",
                    if_exists="append", schema_name=data_conf["alert_db"])
    
def suppress_closed_alert(data_conf, carry_over_open, crime_type, model_version , model_ID, as_of_date, conn):
    ID = data_conf["column_to_preserve"]
    str_IDs_alerts = ','.join(str(v) for v in carry_over_open[ID].to_list())
    alerts_in_db = pd.read_sql(
        f"""select object_id, alert_id, model_version,as_of_date, date_time from {data_conf["alert_db"]}.alerts WHERE object_id in ({str_IDs_alerts}) and crime_type = '{crime_type}' and alert_status = 'closed'""",
            conn)

    alerts_in_db.object_id = alerts_in_db.object_id.astype(int)
    
    if not alerts_in_db.empty:

        for group in alerts_in_db.groupby(['date_time','as_of_date']):
            df_all = carry_over_open.merge(group[1].drop_duplicates(), left_on=ID, right_on = 'object_id', how='left', indicator=True)
            present_acct_number_rows = df_all[df_all['_merge'] == 'both']
            del present_acct_number_rows['_merge']
            str_IDs_common = ','.join(str(v) for v in present_acct_number_rows[ID].to_list())

            absent_acct_number_rows = df_all[df_all['_merge'] == 'left_only']
            del absent_acct_number_rows['_merge']

            old_date_time = str(group[1]["date_time"].values[0])
            time_diff  = datetime.now() - datetime.strptime(old_date_time.split("_")[0], '%d/%m/%Y')

            old_model_version = str(group[1]["model_version"].values[0])
            old_as_of_date = str(group[1]["as_of_date"].values[0])
            old_score_ADS_name = pd.read_sql(f"""select scoring_table from {data_conf["ADS_db_name"]}.BO_metadata where as_of_date = '{old_as_of_date}' and model_version = '{old_model_version}'""", conn).values[0][0]
            old_score_ADS = pd.read_sql(f"""select * from {data_conf["ADS_db_name"]}.{old_score_ADS_name} where {ID} in ({str_IDs_common})""", conn)
            old_as_of_date = str(old_score_ADS[data_conf["Date_col_name"]].values[0])
            old_score_ADS = old_score_ADS.drop(data_conf["Date_col_name"], axis=1)
            old_score_ADS = old_score_ADS.astype(float)
            old_score_ADS = old_score_ADS.sort_values(by=[ID])


            new_score_ADS = pd.read_sql(f"""select * from {data_conf["ADS_db_name"]}.{data_conf["Score_ADS_name"]} where {ID} in ({str_IDs_common})""", conn)
            new_as_of_date = str(new_score_ADS[data_conf["Date_col_name"]].values[0])
            new_score_ADS = new_score_ADS.drop(data_conf["Date_col_name"], axis=1)
            new_score_ADS = new_score_ADS.astype(float)
            new_score_ADS = new_score_ADS.sort_values(by=[ID])

            old_columns = old_score_ADS.columns.to_list()
            new_columns = new_score_ADS.columns.to_list()

            len_common = len(set(old_columns) & set(new_columns))

            similarity_array = 1/(Euclidean_Dist(old_score_ADS, new_score_ADS, new_columns)+1)

            present_acct_number_rows["similarity"] = similarity_array

            l = len(present_acct_number_rows.index) // 2 
            present_acct_number_rows.loc[:l - 1, 'similarity'] = 0.5

            present_acct_number_rows_similar = present_acct_number_rows[present_acct_number_rows["similarity"] >= data_conf["similarity_threshold"]]
            present_acct_number_rows_unsimilar = present_acct_number_rows[present_acct_number_rows["similarity"] <= data_conf["similarity_threshold"]]

            if time_diff.days >= data_conf["recency_threshold"] and not present_acct_number_rows_similar.empty and len_common == max(len(old_columns),len(new_columns)):
                alerts_df = pd.DataFrame(columns=['alert_id', 'alert_status','crime_type','model_id', 'model_version', 'date_time',
                                               'as_of_date', 'resolution_date', 'object_id', 'object_type',
                                               'probability', 'decision_threshold', 'priority', 'last_updated_by'])

                alerts_df['object_id'] = present_acct_number_rows_similar[ID]
                alerts_df["object_type"] = ID
                alerts_df["crime_type"] = crime_type
                alerts_df['date_time'] = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
                alerts_df['as_of_date'] = as_of_date
                alerts_df["alert_id"] = alerts_df["alert_id"].apply(lambda _: uuid.uuid4().hex)
                alerts_df["model_id"] = model_ID
                alerts_df["model_version"] = model_version
                alerts_df["alert_status"] = "suppressed"
                alerts_df["resolution_date"] = None
                alerts_df["probability"] = present_acct_number_rows["Score"]
                alerts_df["decision_threshold"] = data_conf["alert_threshold"]
                alerts_df["priority"] = None
                alerts_df["last_updated_by"] = "system"

                copy_to_sql(alerts_df, table_name="alerts",
                                if_exists="append", schema_name=data_conf["alert_db"])

                supp_alerts_df = pd.DataFrame(columns=['alert_id','parent_alert_id','object_id','object_type','crime_type','suppression_reason', 'mute_start_date',
                                                        'mute_end_date', 'recency_value',
                                                        'similarity_value'])

                supp_alerts_df["alert_id"] = alerts_df["alert_id"]
                supp_alerts_df["parent_alert_id"] = present_acct_number_rows_similar["alert_id"]
                supp_alerts_df['object_id'] = alerts_df['object_id']
                supp_alerts_df["object_type"] = ID
                supp_alerts_df["crime_type"] = crime_type
                supp_alerts_df["mute_start_date"] = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
                supp_alerts_df["mute_end_date"] = None
                supp_alerts_df["suppression_reason"] = "similar and recent"
                supp_alerts_df["recency_value"] = (time_diff.days)
                supp_alerts_df["similarity_value"] = present_acct_number_rows_similar["similarity"]

                copy_to_sql(supp_alerts_df, table_name="alerts_suppressed",
                                if_exists="append", schema_name=data_conf["alert_db"])

                carry_over_closed = present_acct_number_rows_unsimilar.append(absent_acct_number_rows)
                open_alert(data_conf, carry_over_closed, crime_type, model_version , model_ID, as_of_date, conn)
            else:
                carry_over_closed = present_acct_number_rows.append(absent_acct_number_rows)
                open_alert(data_conf, carry_over_closed, crime_type, model_version , model_ID, as_of_date, conn)
    else:
        open_alert(data_conf, carry_over_open, crime_type, model_version , model_ID, as_of_date, conn)
        

def generate_alert(data_conf, alert_ads, crime_type, model_version , model_ID, as_of_date, conn):  # local_exp contains scoring results
    create_tables(data_conf)
    carry_over_custom = suppress_custom_alert(data_conf, alert_ads, crime_type, model_version , model_ID, as_of_date, conn)
    if not carry_over_custom.empty:
        carry_over_open = suppress_open_alert(data_conf, carry_over_custom, crime_type, model_version , model_ID, as_of_date, conn)
        if not carry_over_open.empty:
            suppress_closed_alert(data_conf, carry_over_open, crime_type, model_version , model_ID, as_of_date, conn)


def evaluate(data_conf, model_conf, **kwargs):
    # Establish database connection
    eng, conn = create_connection(data_conf)
    model_version = kwargs.get("model_version")
    model_ID = kwargs.get("model_id")

    features_data, labels = get_test_data(data_conf, conn)

    model_df = pd.read_sql(
        f"""select model from {data_conf["datascience_db"]}.models_artifacts_BO where model_version='{model_version}'""",
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

    # dump results as json file evaluation.json to models/ folder
    with open("models/evaluation.json", "w+") as f:
        json.dump(scores, f)

    table_name = f"""Evaluation_{data_conf["Evaluation_ADS_name"]}_Mv_{model_version}_M_{model_ID}"""

    copy_to_sql(df=eval_df, table_name=table_name,
                schema_name=data_conf["datascience_db"],
                temporary=False,
                if_exists='replace')

    conn.execute(f"""
                UPDATE {data_conf["datascience_db"]}.global_explainability_metadata SET evaluation_table='{table_name}' WHERE model_version='{model_version}'; 
                """)

    print("Evaluation complete...")


def score(data_conf, model_conf, db=None, deploy=True, **kwargs):
    print("Scoring started..")

    #epoch_timestamp = round(time.time())
    timestamp = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
    conn, eng = create_connection(data_conf)
    model_version = kwargs.get("model_version")
    query = f"""select model_id from {data_conf["ADS_db_name"]}.models_artifacts_BO WHERE 
    model_version='{model_version}'"""

    model_ID = pd.read_sql_query(query, conn)["model_id"].to_string(index=False).replace(" ", "")

    # Query to execute.
    ivsm_scoring_query = f"""select * from adldemo_ivsm.IVSM_score2(
          on {data_conf["ADS_db_name"]}.{data_conf["Score_ADS_name"]}
          on (select model_id, model from {data_conf["ADS_db_name"]}.models_artifacts_BO WHERE model_version='{model_version}') DIMENSION 
          using
              ModelID('{model_ID}')
              ColumnsToPreserve('{data_conf["column_to_preserve"]}') 
              ModelType('PMML')
              ModelSpecificSettings('PMML_OUTPUT_TYPE=ALL')
      ) sc;
      """

    print("Start scoring...")

    scoring_result = pd.read_sql_query(ivsm_scoring_query, conn)
    prob_1 = scoring_result["score_result"].apply(lambda x: json.loads(x)["probability(1)"]).astype(float)

    scoring_result.insert(2, "Score", prob_1)
    scoring_result.insert(3, "probability_N", 1 - scoring_result["Score"])
    prediction = scoring_result["Score"].apply(get_prediction,
                                                       args=({model_conf["hyperParameters"]["probability_threshold"]}))
    scoring_result.insert(4, "prediction", prediction)
    scoring_result.insert(5, "model_version", model_version)

    scoring_table_name = f"""Scoring_{data_conf["Score_ADS_name"]}_Mv_{model_version}_M_{model_ID}_{timestamp}"""
    
    copy_to_sql(df=scoring_result, table_name=scoring_table_name, schema_name=data_conf["datascience_db"], temporary=False,
                if_exists='replace')

    print("Starting Explaining Model")

    as_of_date, alert_ads = explainability(scoring_result, data_conf, model_version, model_ID, conn, timestamp)
    
    process_local_explainability_metadata(model_version, model_ID, data_conf, as_of_date,timestamp)
    
    crime_type = 'Bustout'
    generate_alert(data_conf, alert_ads, crime_type, model_version , model_ID, as_of_date, conn)
    
    logging.info("End scoring")

    print("End scoring")


# if __name__ == "__main__":
#     try:
#         data_conf = {
#             "hostname": "tdprd2.td.teradata.com",
#             "ADS_db_name": "fincrime_bustout_dev",
#             "Score_ADS_name": "BO_Scoring_ADS_1",
#             "column_to_preserve": "acct_no",
#             "Date_col_name": "fc_agg_summary_date",
#             "local_explainabilty_threshold": 0.5,
#             "datascience_db": "fincrime_bustout_dev"
#         }
#
#         with open("../config.json") as json_file:
#             model_conf = json.load(json_file)
#
#         with open("../cred.json") as json_file:
#             creds = json.load(json_file)
#
#         os.environ["TD_USERNAME"] = creds["TD_Username"]
#         os.environ["TD_PASSWORD"] = creds["TD_Password"]
#
#         score(data_conf, model_conf, model_version='123', model_id='123')
#
#     except Exception as e:
#         print(e)


