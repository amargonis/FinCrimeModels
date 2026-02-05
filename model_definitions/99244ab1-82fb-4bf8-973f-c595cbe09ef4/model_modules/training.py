import os
import pandas as pd
import teradataml as tdml
import json

from sklearn2pmml import sklearn2pmml
from teradataml.context.context import *
from teradataml.dataframe.copy_to import copy_to_sql

import lightgbm as lgb
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split
from sklearn2pmml.decoration import ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from pypmml import Model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def input_table_sql(data_conf, num_feat_str, cat_feat_str, label_feat, inst_id_feat):
    """
    Prepare a sql statement to specify input table for DecisionForest On clause.
    Training set selection.

    :param data_conf: (dict): The data set metadata
    :param num_feat_str: (str): sql string for numerical features
    :param cat_feat_str: (str): sql string for categorical features
    :param label_feat: (str): label column name
    :param inst_id_feat: (str): instance id
    :return: train_select: (str): sql statement to specify decision forest input table
    """
    train_select = f"""
    SELECT {num_feat_str}
    {cat_feat_str}
    pty.{label_feat} as Target
    FROM {data_conf['src_db']}.{data_conf['party_info']} pty
    INNER JOIN {data_conf['features_db']}.{data_conf['dataset']} agg
    ON pty.partyid = agg.partyid
    INNER JOIN {data_conf['src_db']}.{data_conf['train_test_split']} inst
    ON pty.{inst_id_feat} = inst.inst_id
    WHERE inst.train=1
    """
    return train_select


def get_test_data_query(data_conf, num_feat_str, cat_feat_str, label_feat, inst_id_feat):
    """
    Prepare a sql statement to specify input table for DecisionForest On clause.
    Training set selection.

    :param data_conf: (dict): The data set metadata
    :param num_feat_str: (str): sql string for numerical features
    :param cat_feat_str: (str): sql string for categorical features
    :param label_feat: (str): label column name
    :param inst_id_feat: (str): instance id
    :return: train_select: (str): sql statement to specify decision forest input table
    """
    test_select = f"""
    SELECT {num_feat_str}
    {cat_feat_str}
    pty.{label_feat} as Target
    FROM {data_conf['src_db']}.{data_conf['party_info']} pty
    INNER JOIN {data_conf['features_db']}.{data_conf['dataset']} agg
    ON pty.partyid = agg.partyid
    INNER JOIN {data_conf['src_db']}.{data_conf['train_test_split']} inst
    ON pty.{inst_id_feat} = inst.inst_id
    WHERE inst.train=1
    """
    return test_select


def if_exists_drop_tbl(db_name, tbl_name, conn):
    ret = pd.read_sql(
        f"select distinct TableName FROM DBC.TablesV WHERE DatabaseName= '{db_name}' and TableName='{tbl_name}'", conn)
    if len(ret) > 0:
        conn.execute(f"DROP TABLE {db_name}.{tbl_name}")


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


def build_feat_str(r):
    """
    To generate a sql statement for provided feature

    :param r: (series): feature name
    """
    st = r.feature
    if r.source == 'party':
        return 'pty' + '.' + st + ' as feat' + str(r.feature_id)
    else:
        return 'agg' + '.' + st + ' as feat' + str(r.feature_id)


def select_features(data_conf, eng):
    """
    Join features information based on selected model features

    :param data_conf: (dict): dataset metadata
    :param eng: (object): Database engine
    """
    model_features = pd.read_sql_query(f"sel * from {data_conf['db_name']}.{data_conf['model_features']}", eng)
    feature_metadata = pd.read_sql_query(f"sel * from {data_conf['db_name']}.{data_conf['feature_metadata']}", eng)

    features = model_features[model_features.model_id == data_conf['mdl_id']].merge(feature_metadata, on='feature_id')

    return features


def numerical_features(features):
    """
    # prepare a concatenated sql query string for numerical features

    :param features: (dataframe): all features
    """
    num_inps = ''
    num_feat_str = ''
    if not features[(features.status == 'feat') & (features.ds_type == 'n')].empty:
        num_feat_str = '\n,'.join(features[(features.status == 'feat') & (features.ds_type == 'n')][
                                      ['feature', 'source', 'feature_id']].apply(build_feat_str, axis=1))

        for i in num_feat_str.split('\n,'):
            num_inps += "'" + i.split(' as ')[1] + "',"
        num_inps = num_inps.rstrip(",")
        num_feat_str = num_feat_str + '\n,'

    return num_feat_str, num_inps


def categorical_features(features):
    """
    # prepare a concatenated sql query string for categorical features

    :param features: (dataframe): all features
    """
    cat_inps = ''
    cat_feat_str = ''
    if not features[(features.status == 'feat') & (features.ds_type == 'c')].empty:
        cat_feat_str = '\n,'.join(features[(features.status == 'feat') & (features.ds_type == 'c')][
                                      ['feature', 'source', 'feature_id']].apply(build_feat_str, axis=1))

        for i in cat_feat_str.split('\n,'):
            cat_inps += "'" + i.split(' as ')[1] + "',"
        cat_inps = cat_inps.rstrip(",")
        cat_feat_str = cat_feat_str + '\n,'

    return cat_feat_str, cat_inps


def get_id_feature(features):
    """
    # select instance id column

    :param features: (dataframe): all features
    """
    inst_id_feat = features[features.status == 'id'].feature.values[0]
    return inst_id_feat


def get_label(features):
    """
    # select label column

    :param features: (dataframe): all features
    """
    label_feat = features[features.status == 'label'].feature.values[0]
    return label_feat


def get_training_data(query, eng):
    return (pd.read_sql_query(query, eng))  # execute training data query


def get_test_data(query, eng):
    return (pd.read_sql_query(query, eng))  # execute training data query


def evaluation(y_true, y_pred):
    # model = Model.fromFile('LightGBMAudit.pmml')
    # result = model.predict(training_data)
    # print(result)

    acc_score = accuracy_score(y_true, y_pred, normalize=False)
    print(acc_score)
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(conf_matrix)


def delete_record_if_exists(db_name, model_table, model_version, conn):
    ret = pd.read_sql_query(f"select * from {db_name}.{model_table} WHERE model_version='{model_version}'", conn)

    if len(ret) > 0:
        conn.execute("delete from " + db_name + "." + model_table + " WHERE model_version='" + model_version + "'")
        print("model deleted")


def loadModel(db_name, model_table, conn, model_id, model_version):
    if_not_exist_create_table(db_name, model_table, conn)
    delete_record_if_exists(db_name,model_table, model_version, conn)

    model_bytes = open("models/model.pmml", "rb").read()

    conn.execute(f"insert into " + db_name + "." + model_table + "(model_id, model, model_version) values(?,?,?)",
                 model_id, model_bytes, model_version)

    # Table name as AOA project name.

    # Table name as AOA project name.
    # Model status dump in the table (deployed or not)
    # Time stamp dump.
    # CHange data config vars in code.
    print("model dump done.")


# Final Commit before code refactoring
# add **kwargs as third argument
def train(data_conf, model_conf, **kwargs):
    """
    Training method that is called by AnalyticOps framework

    :param data_conf: (dict): The data set metadata
    :param model_conf: (dict): The model configuration to use
    :param kwargs: (optional) not used
    :return:
    """

    model_version = kwargs.get("model_version")
    model_id = kwargs.get("model_id")

    # model_version = "ModelVersion1"
    # model_id = "1"

    print("Model version is ")
    print(model_version)


    hparam = model_conf['hyperParameters']

    # Establish database connection
    eng, conn = create_connection(data_conf)
    #
    # load features
    mrg_feat = select_features(data_conf, eng)

    # prepare a sql string and list of numeric features
    num_feat_str, num_inps = numerical_features(mrg_feat)

    # prepare a sql string and list of categorical features
    cat_feat_str, cat_inps = categorical_features(mrg_feat)

    # get instance id
    inst_id_feat = get_id_feature(mrg_feat)

    # load label feature
    label_feat = get_label(mrg_feat)

    # Develop SQL select statement for Decision Forest ON statement
    train_select = input_table_sql(data_conf, num_feat_str, cat_feat_str, label_feat, inst_id_feat)
    training_data = get_training_data(train_select, eng)

    test_select = get_test_data_query(data_conf, num_feat_str, cat_feat_str, label_feat, inst_id_feat)
    test_data = get_test_data(test_select, eng)

    # copy_to_sql(df=test_data, table_name="iVSM_test_data_db", schema_name="fincrime_aml_dev", temporary=False,
    #             if_exists='replace')

    # Separating labels and training data
    training_Y = training_data["Target"]
    training_X = training_data.drop("Target", axis=1)

    column_names = training_X.columns.values.tolist()

    mapper = DataFrameMapper([
        (column_names, [ContinuousDomain()])
    ])

    # train
    gbm_classifier = lgb.LGBMClassifier(num_leaves=hparam["num_leaves"],
                                        learning_rate=hparam["learning_rate"],
                                        n_estimators=hparam["n_estimators"])
    pipeline = PMMLPipeline([
        ("mapper", mapper),
        ("classifier", gbm_classifier)
    ])

    pipeline.fit(training_X, training_Y)

    # Export to PMML File
    sklearn2pmml(pipeline, "models/model.pmml", with_repr=True)

    # Uplaod model to Vantage
    loadModel(model_conf['db_name'], model_conf['mdl_tbl'], conn, model_id, model_version)  # Load model to db
