import json

import pandas as pd
import numpy as np
from sklearn2pmml import sklearn2pmml
from sklearn.model_selection import train_test_split
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.ensemble import RandomForestClassifier
from teradataml.dataframe.copy_to import copy_to_sql
import pickle, shap, os, joblib
from teradataml.context.context import *


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


def create_base(data_conf, conn):
    meta_data = pd.read_sql(
        f"""select model_version,model_id,feature,ml_type from {data_conf["metadata_db_name"]}.{data_conf["metadata_dataset_name"]} where model_id={data_conf["dataset_ID"]} and model_version = {data_conf["dataset_version"]}""",
        conn)
    features = ",".join(meta_data[meta_data['ml_type'] == 2]['feature'].tolist())
    ID = meta_data[meta_data['ml_type'] == 1]['feature'].values[0]
    label = meta_data[meta_data['ml_type'] == 3]['feature'].values[0]
    dataset_version = meta_data['model_version'].values[0]
    dataset_ID = meta_data['model_id'].values[0]

    query = f"""Create multiset table {data_conf["ADS_db_name"]}.base_Dv_{dataset_version}_D_{dataset_ID} as (
                        select distinct {ID},{label} from {data_conf["ADS_db_name"]}.{data_conf["ADS_name"]}
                        ) with data primary index({ID},{label})"""

    try:
        conn.execute(f"""Drop table {data_conf["ADS_db_name"]}.base_Dv_{dataset_version}_D_{dataset_ID}""")
        conn.execute(query)
    except Exception as e:
        print(e)
        conn.execute(query)

    print("base_created")
    return ID, label, features, dataset_version, dataset_ID


def create_sampled_base(data_conf, ID, label, features, dataset_version, dataset_ID, conn):
    BO_accs = pd.read_sql(
        f"""select acct_no from {data_conf["ADS_db_name"]}.base_Dv_{dataset_version}_D_{dataset_ID} where {label} = 'Y'""",
        conn)
    count_BO = len(BO_accs.index)
    count_NBO = int((count_BO / data_conf["sampling_rate"]) - count_BO)
    NBO_accs = pd.read_sql(
        f"""select {ID} from {data_conf["ADS_db_name"]}.base_Dv_{dataset_version}_D_{dataset_ID} where {label} = 'N' sample {count_NBO}""",
        conn)
    sample_accs = pd.concat([NBO_accs, BO_accs])
    copy_to_sql(sample_accs, table_name=f"""sample_accs_Dv_{dataset_version}_D_{dataset_ID}""", if_exists="replace",
                schema_name=data_conf["ADS_db_name"])


def create_sampled_ADS(data_conf, ID, dataset_version, dataset_ID, conn):
    query = f"""Create set table {data_conf["ADS_db_name"]}.sampled_Dv_{dataset_version}_D_{dataset_ID} as (
    select ADS.*
    FROM {data_conf["ADS_db_name"]}.{data_conf["ADS_name"]} ADS
    JOIN 
    {data_conf["ADS_db_name"]}.sample_accs_Dv_{dataset_version}_D_{dataset_ID} Accs
    on ADS.{ID} = Accs.{ID}
    ) with data primary index({ID})"""
    try:
        conn.execute(f"""Drop table {data_conf["ADS_db_name"]}.sampled_Dv_{dataset_version}_D_{dataset_ID}""")
        conn.execute(query)
    except Exception as e:
        print(e)
        conn.execute(query)

    print("sampled_ADS_created")
    return f"""sampled_Dv_{dataset_version}_D_{dataset_ID}"""


def create_ADS(data_conf, src_data, ID, label, features, dataset_version, dataset_ID, conn):
    query = f"""create multiset table {data_conf["ADS_db_name"]}.ATO_ADS_Dv_{dataset_version}_D_{dataset_ID} as (
                select {ID},{data_conf["Date_col_name"]},{features},{label} as fraud
            from {data_conf["ADS_db_name"]}.{src_data}) with data primary index({ID})"""

    try:
        conn.execute(f"""Drop table {data_conf["ADS_db_name"]}.ATO_ADS_Dv_{dataset_version}_D_{dataset_ID}""")
        conn.execute(query)
    except Exception as e:
        print(e)
        conn.execute(query)


def get_train_data(data_conf, ID, label, conn, dataset_version, dataset_ID):
    X = pd.read_sql(f"""select * from {data_conf["ADS_db_name"]}.ATO_ADS_Dv_{dataset_version}_D_{dataset_ID}""", conn)
    #X_score = X[X['fraud'].isna()]
    X = X.dropna()
    X = X.drop_duplicates()
    #X_score = X_score.drop(["fraud"], axis=1)
    X.fraud.replace(('Y', 'N'), (1, 0), regex=True, inplace=True)
    #X_score = X[X.groupby('acct_no')[data_conf["Date_col_name"]].transform('max') == X[data_conf["Date_col_name"]]]
    #X_score = X_score.drop([label, "fraud"], axis=1)
    #X = X.drop(X[X.groupby('acct_no')[data_conf["Date_col_name"]].transform('max') == X[data_conf["Date_col_name"]]].index)
    y = X["fraud"]
    X = X.drop([data_conf["Date_col_name"], "fraud"], axis=1)
    for column in X.select_dtypes(include='object').columns:
        X = pd.concat([X.drop(column, axis=1), pd.get_dummies(X[column], prefix=column)], axis=1)
    X = X.astype(float)
    X = X.set_index(ID)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=data_conf['test_size'])

    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(level=0, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    Xy_test = pd.concat([X_test, y_test], axis=1)

    copy_to_sql(Xy_test, table_name=data_conf["Test_ADS_name"], if_exists="replace",
                schema_name=data_conf["Test_ADS_db_name"])
    #copy_to_sql(X_score, table_name=data_conf["Score_ADS_name"], if_exists="replace",
                #schema_name=data_conf["Score_ADS_db_name"])
    return X_train, y_train


def train_model(X_train, y_train, data_conf, model_conf, dataset_version, dataset_ID, model_version, model_ID, conn):
    hyperparams = model_conf["hyperParameters"]

    skclassifier = RandomForestClassifier(n_estimators=hyperparams["n_estimators"],
                                          max_depth=hyperparams["max_depth"],
                                          min_samples_split=hyperparams["min_samples_split"],
                                          min_samples_leaf=hyperparams["min_samples_leaf"],
                                          max_features=hyperparams["max_features"],
                                          min_impurity_decrease=hyperparams["min_impurity_decrease"],
                                          bootstrap=hyperparams["bootstrap"],
                                          oob_score=hyperparams["oob_score"],
                                          verbose=hyperparams["verbose"],
                                          warm_start=hyperparams["warm_start"]
                                          )

    classifier = PMMLPipeline([('classifier', skclassifier)])

    # fit model to training data
    classifier.fit(X_train, y_train)
    skclassifier.fit(X_train, y_train)
    
    joblib.dump(skclassifier, "artifacts/output/model.joblib")

    feats = {}  # a dict to hold feature_name: feature_importance
    for feature, importance in zip(X_train.columns, skclassifier.feature_importances_):
        feats[feature] = importance

    importances = pd.DataFrame(feats.items(), columns=['Feature', 'Feature_Importance'])
    print("Finished training")

#     explainer = shap.Explainer(classifier.predict,X_train)

#     shap_values = explainer(X_train)
#     shap_v = pd.DataFrame(shap_values.values)
#     shap_v.columns = X_train.columns
#     #shap_abs = np.abs(shap_v)
#     k=pd.DataFrame(shap_v.mean()).reset_index()
#     k.columns = ['Variable','Feature_importance']
    #k = k.sort_values(by='SHAP',ascending = True)
    copy_to_sql(importances,
                table_name=f"""Global_Interpretability_Mv_{model_version}_Mid_{model_ID}_Dv_{dataset_version}_Did_{dataset_ID}""",
                if_exists="replace", schema_name=data_conf["datascience_db"])

    # global explanabilit metadata
    g_exp_metadata = pd.DataFrame(
        columns=["model_version", "model_id", "dataset_version", "dataset_id", "database_name",
                 "global_explainability_table", "evaluation_table", "model_type"])

    g_exp_metadata = g_exp_metadata.append({"model_version": model_version,
                                            "model_id": model_ID,
                                            "dataset_version": dataset_version,
                                            "dataset_id": dataset_ID,
                                            "database_name": data_conf["datascience_db"],
                                            "global_explainability_table": f"""Global_Interpretability_Mv_{model_version}_Mid_{model_ID}_Dv_{dataset_version}_Did_{dataset_ID}""",
                                            "evaluation_table": None,
                                            "model_type": "RandomForestClassifier"
                                            }, ignore_index=True)

    copy_to_sql(df=g_exp_metadata, table_name="global_explainability_metadata",
                schema_name=data_conf["datascience_db"],
                temporary=False,
                if_exists='append')

    # export model artefacts to models/ folder
    if not os.path.exists('models'):
        os.makedirs('models')
    sklearn2pmml(classifier, f"""models/Modelv_{model_version}_{model_ID}.pmml""", with_repr=True)
    print("Saved trained model")
    return skclassifier


def delete_record_if_exists(db_name, model_version, conn):
    conn.execute(f"""delete from {db_name}.models_artifacts_ATO WHERE model_version='{model_version}'""")


def if_not_exist_create_table(db_name, conn):
    ret = pd.read_sql(
        f"select distinct TableName FROM DBC.TablesV WHERE DatabaseName= '{db_name}' and TableName='models_artifacts_ATO'",
        conn)

    if not len(ret) > 0:
        conn.execute(f"""CREATE MULTISET TABLE {db_name}.models_artifacts_ATO, FALLBACK , NO BEFORE JOURNAL,
                        NO AFTER JOURNAL, CHECKSUM = DEFAULT, DEFAULT MERGEBLOCKRATIO, MAP = TD_MAP1
                        (
                        model_version VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC,
                        model_id VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC,
                        deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        model BLOB(2097088000),
                        skmodel BLOB(2097088000),
                        dataset_version VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC,
                        dataset_id VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC
                        )
                        UNIQUE PRIMARY INDEX ( model_version );""")


# Create metadata table for holding model inference attributes.
def create_models_metadata_table(db_name, conn):
    ret = pd.read_sql(
        f"select distinct TableName FROM DBC.TablesV WHERE DatabaseName= '{db_name}' and TableName='models_metadata'",
        conn)

    if not len(ret) > 0:
        conn.execute(f"""CREATE MULTISET TABLE {db_name}.models_metadata, FALLBACK , NO BEFORE JOURNAL,
                        NO AFTER JOURNAL, CHECKSUM = DEFAULT, DEFAULT MERGEBLOCKRATIO, MAP = TD_MAP1
                        (
                        model_version VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC,
                        model_id VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC,
                        dataset_version VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC,
                        dataset_id VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC,
                        global_explanability_table VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC,
                        evaluation_table VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC,
                        local_explanability_table VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC,
                        scoring_table VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC,
                        model_type VARCHAR(255) CHARACTER SET LATIN CASESPECIFIC,
                        model_status,
                        deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                        UNIQUE PRIMARY INDEX ( model_version );""")


def loadModel(data_conf, classifier, dataset_version, dataset_ID, model_version, model_ID, conn):
    if_not_exist_create_table(data_conf["datascience_db"], conn)
    delete_record_if_exists(data_conf["datascience_db"], str(model_version), conn)

    pmml_bytes = open(f"""models/Modelv_{model_version}_{model_ID}.pmml""", "rb").read()

    classifier_bytes = pickle.dumps(classifier)
    conn.execute(
        f"""insert into {data_conf["datascience_db"]}.models_artifacts_ATO(model_version, model_id, model, skmodel, 
dataset_version, dataset_ID) values(?,?,?,?,?,?)""",
        str(model_version), str(model_ID), pmml_bytes, classifier_bytes, str(dataset_version), str(dataset_ID))

    print("model dump done.")


def train(data_conf, model_conf, **kwargs):
    eng, conn = create_connection(data_conf)

    model_version = kwargs.get("model_version")
    model_id = kwargs.get("model_id")

    if data_conf["sampling"]:
        ID, label, features, dataset_version, dataset_ID = create_base(data_conf, conn)
        create_sampled_base(data_conf, ID, label, features, dataset_version, dataset_ID, conn)
        src_data = create_sampled_ADS(data_conf, ID, dataset_version, dataset_ID, conn)
        create_ADS(data_conf, src_data, ID, label, features, dataset_version, dataset_ID, conn)
        X_train, y_train = get_train_data(data_conf, ID, label, conn, dataset_version, dataset_ID)
        classifier = train_model(X_train, y_train, data_conf, model_conf, dataset_version, dataset_ID, model_version,
                                 model_id, conn)
        loadModel(data_conf, classifier, dataset_version, dataset_ID, model_version, model_id, conn)
    else:
        ID, label, features, dataset_version, dataset_ID = create_base(data_conf, conn)
        create_ADS(data_conf, data_conf["ADS_name"], ID, label, features, dataset_version, dataset_ID, conn)
        X_train, y_train = get_train_data(data_conf, ID, label, conn)
        classifier = train_model(X_train, y_train, data_conf, model_conf, dataset_version, dataset_ID, model_version,
                                 model_id, conn)
        loadModel(data_conf, classifier, dataset_version, dataset_ID, model_version, model_id, conn)
