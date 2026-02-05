import json
import pandas as pd
from teradataml.context.context import *
from teradataml.dataframe.copy_to import copy_to_sql
from pypmml import Model
from sklearn import metrics
import logging


# from . import db_common
# from . import feature_selection


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


def if_exists_drop_tbl(db_name, tbl_name, conn):
    ret = pd.read_sql(
        f"select distinct TableName FROM DBC.TablesV WHERE DatabaseName= '{db_name}' and TableName='{tbl_name}'", conn)
    if len(ret) > 0:
        conn.execute(f"DROP TABLE {db_name}.{tbl_name}")


def get_connection(data_conf):
    host = data_conf["hostname"]
    username = os.environ["TD_USERNAME"]
    password = os.environ["TD_PASSWORD"]
    logmech = os.getenv("TD_LOGMECH", "TDNEGO")

    eng = create_context(host=host, username=username, password=password, logmech=logmech)
    conn = eng.connect()

    return conn, eng


def get_test_data(conn, eng, data_conf):
    model_features = pd.read_sql_query(f"sel * from {data_conf['db_name']}.{data_conf['model_features']}", eng)
    feature_metadata = pd.read_sql_query(f"sel * from {data_conf['db_name']}.{data_conf['feature_metadata']}", eng)

    mrg_feat = model_features[model_features.model_id == data_conf['mdl_id']].merge(feature_metadata, on='feature_id')

    NumInps = ''
    num_feat_str = ''
    if not mrg_feat[(mrg_feat.status == 'feat') & (mrg_feat.ds_type == 'n')].empty:
        num_feat_str = '\n,'.join(mrg_feat[(mrg_feat.status == 'feat') & (mrg_feat.ds_type == 'n')][
                                      ['feature', 'source', 'feature_id']].apply(build_feat_str, axis=1))

        for i in num_feat_str.split('\n,'):
            NumInps += "'" + i.split(' as ')[1] + "',"
        NumInps = NumInps.rstrip(",")
        num_feat_str = num_feat_str + '\n,'

    CatInps = ''
    cat_feat_str = ''
    if not mrg_feat[(mrg_feat.status == 'feat') & (mrg_feat.ds_type == 'c')].empty:
        cat_feat_str = '\n,'.join(mrg_feat[(mrg_feat.status == 'feat') & (mrg_feat.ds_type == 'c')][
                                      ['feature', 'source', 'feature_id']].apply(build_feat_str, axis=1))

        for i in cat_feat_str.split('\n,'):
            CatInps += "'" + i.split(' as ')[1] + "',"
        CatInps = CatInps.rstrip(",")
        cat_feat_str = cat_feat_str + '\n,'

    inst_id_feat = mrg_feat[mrg_feat.status == 'id'].feature.values[0]
    label_feat = "pty." + mrg_feat[mrg_feat.status == 'label'].feature.values[0]

    test_select = f"""
       SELECT {num_feat_str}
       {cat_feat_str}
       {label_feat} as Target
       ,pty.{inst_id_feat} as Inst_Id
       FROM {data_conf['src_db']}.{data_conf['party_info']} pty
       INNER JOIN {data_conf['features_db']}.{data_conf['dataset']} agg
       ON pty.partyid = agg.partyid
       INNER JOIN {data_conf['src_db']}.{data_conf['train_test_split']} inst
       ON pty.{inst_id_feat} = inst.inst_id
       WHERE inst.train=0
       """

    return test_select


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

    conn, eng = get_connection(data_conf)
    test_select = get_test_data(conn, eng, data_conf)

    model_version = kwargs.get("model_version")

    print("Model Version: ")
    print(model_version)

    # Get model ID
    query = f"""select model_id from {model_conf['db_name']}.{model_conf['mdl_tbl']} WHERE model_version='{model_version}'"""
    model_id = pd.read_sql_query(query, conn)["model_id"].to_string(index=False).replace(" ", "")

    # Query to execute.
    ivsm_scoring_query = f"""
      select * from adldemo_ivsm.IVSM_score2(
          on ({test_select})
          on (select model_id, model from {model_conf['db_name']}.{model_conf['mdl_tbl']} WHERE model_version='{model_version}') DIMENSION 
          using
              ModelID('{model_id}')
              ColumnsToPreserve('{model_conf["column_to_preserve"]}') 
              ModelType('PMML')
              ModelSpecificSettings('PMML_OUTPUT_TYPE=ALL')
      ) sc;
      """

    predict_table = data_conf['predict_tbl'] + '_' + model_version

    print("Start scoring...")

    scoring_result = pd.read_sql_query(ivsm_scoring_query, conn)
    prob_1 = scoring_result["score_result"].apply(lambda x: json.loads(x)["probability(1)"]).astype(float)

    scoring_result.insert(2, "probability_1", prob_1)
    scoring_result.insert(3, "probability_0", 1 - scoring_result["probability_1"])
    prediction = scoring_result["probability_1"].apply(get_prediction,
                                                       args=({model_conf["hyperParameters"]["probability_threshold"]}))
    scoring_result.insert(4, "prediction", prediction)

    copy_to_sql(df=scoring_result, table_name=predict_table, schema_name=data_conf["db_name"], temporary=False,
                if_exists='replace')

    print("End scoring")
    logging.info("End scoring")


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

    print(data_conf)

    host = data_conf["hostname"]

    username = os.environ["TD_USERNAME"]
    password = os.environ["TD_PASSWORD"]

    # username = os.environ["TD_USERNAME"]
    # password = os.environ["TD_PASSWORD"]
    logmech = os.getenv("TD_LOGMECH", "TDNEGO")

    eng = create_context(host=host, username=username, password=password, logmech=logmech)
    conn = eng.connect()

    model_features = pd.read_sql_query(f"sel * from {data_conf['db_name']}.{data_conf['model_features']}", eng)
    feature_metadata = pd.read_sql_query(f"sel * from {data_conf['db_name']}.{data_conf['feature_metadata']}", eng)

    mrg_feat = model_features[model_features.model_id == data_conf['mdl_id']].merge(feature_metadata, on='feature_id')

    NumInps = ''
    num_feat_str = ''
    if not mrg_feat[(mrg_feat.status == 'feat') & (mrg_feat.ds_type == 'n')].empty:
        num_feat_str = '\n,'.join(mrg_feat[(mrg_feat.status == 'feat') & (mrg_feat.ds_type == 'n')][
                                      ['feature', 'source', 'feature_id']].apply(build_feat_str, axis=1))

        for i in num_feat_str.split('\n,'):
            NumInps += "'" + i.split(' as ')[1] + "',"
        NumInps = NumInps.rstrip(",")
        num_feat_str = num_feat_str + '\n,'

    CatInps = ''
    cat_feat_str = ''
    if not mrg_feat[(mrg_feat.status == 'feat') & (mrg_feat.ds_type == 'c')].empty:
        cat_feat_str = '\n,'.join(mrg_feat[(mrg_feat.status == 'feat') & (mrg_feat.ds_type == 'c')][
                                      ['feature', 'source', 'feature_id']].apply(build_feat_str, axis=1))

        for i in cat_feat_str.split('\n,'):
            CatInps += "'" + i.split(' as ')[1] + "',"
        CatInps = CatInps.rstrip(",")
        cat_feat_str = cat_feat_str + '\n,'

    inst_id_feat = mrg_feat[mrg_feat.status == 'id'].feature.values[0]
    label_feat = "pty." + mrg_feat[mrg_feat.status == 'label'].feature.values[0]

    test_select = f"""
    SELECT {num_feat_str}
    {cat_feat_str}
    {label_feat} as Target
    ,pty.{inst_id_feat} as Inst_Id
    FROM {data_conf['src_db']}.{data_conf['party_info']} pty
    INNER JOIN {data_conf['features_db']}.{data_conf['dataset']} agg
    ON pty.partyid = agg.partyid
    INNER JOIN {data_conf['src_db']}.{data_conf['train_test_split']} inst
    ON pty.{inst_id_feat} = inst.inst_id
    WHERE inst.train=0
    """
    test_data = pd.read_sql_query(test_select, eng)
    test_X = test_data.drop(test_data.columns[-1], axis=1)

    model = Model.load('models/model.pmml')
    score_y = model.predict(test_X)
    print("Following are the model evaluation results. ")
    print(score_y)

    probability_score = score_y["probability(1)"]
    prediction = {}

    # Get ground truth
    test_y = test_data.iloc[:, -2]

    for i in range(probability_score.size):
        if probability_score[i] >= 0.5:
            prediction[i] = 1
        else:
            prediction[i] = 0

    # Calculate ROC Curve
    fpr, tpr, thresholds = metrics.roc_curve(test_y, score_y["probability(1)"])
    auc = metrics.auc(fpr, tpr)
    GINI = (2 * auc) - 1

    if_exists_drop_tbl(data_conf['db_name'], data_conf['predict_tbl'], conn)
    if_exists_drop_tbl(data_conf['db_name'], data_conf['evaluation_tbl'], conn)

    predict_frame = pd.DataFrame()

    predict_frame["Inst_Id"] = test_data.iloc[:, -1]
    predict_frame["target"] = test_y
    predict_frame["prediction"] = prediction.values()
    predict_frame["probability"] = probability_score
    print(predict_frame.head())

    conn.execute(f"""CREATE TABLE {data_conf['db_name']}.{data_conf['predict_tbl']} (Inst_Id BIGINT,
    target BIGINT,
	prediction BIGINT,
	probability FLOAT
    );""")

    conn.execute(f"""CREATE TABLE {data_conf['db_name']}.{data_conf['evaluation_tbl']} (auc FLOAT,
    	gini FLOAT
        );""")

    conn.execute(f"""INSERT INTO {data_conf['db_name']}.{data_conf['evaluation_tbl']}
    (auc, gini) VALUES({auc}, {GINI});
    """)

    # for index, row in predict_frame.iterrows():
    #     print(row)
    #     conn.execute(f"""INSERT INTO {data_conf['db_name']}.{data_conf['predict_tbl']}
    #        (Inst_Id, target, prediction, probability) VALUES(?,?,?,?);
    #        """, row["Inst_Id"], row["target"], row["prediction"], row["probability"])

    copy_to_sql(df=predict_frame, table_name=data_conf['predict_tbl'], schema_name=data_conf["db_name"],
                temporary=False,
                if_exists='replace')

    scores = {}
    scores['auc'] = auc
    scores['gini'] = GINI

    # dump results as json file evaluation.json to models/ folder
    with open("models/evaluation.json", "w+") as f:
        json.dump(scores, f)
    print("Evaluation complete...")

    print("Evaluation Results pushed")

    # predict_df = pd.read_csv(data_conf['location'])
    # features = 'sepallength,sepalwidth,petallength,petalwidth'.split(',')
    # X_predict = predict_df.loc[:, features]
    # y_test = predict_df['class']
    # knn = joblib.load('models/iris_knn.joblib')
    #
    # y_predict = knn.predict(X_predict)
    # scores = {}
    # scores['accuracy'] = metrics.accuracy_score(y_test, y_predict)
    # print("model accuracy is ", scores['accuracy'])
    #
    # # dump results as json file evaluation.json to models/ folder
    # with open("models/evaluation.json", "w+") as f:
    #     json.dump(scores, f)
    # print("Evaluation complete...")



