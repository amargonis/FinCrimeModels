import uuid

import pandas as pd
from teradataml.dataframe.copy_to import copy_to_sql

def create_tables(db_name, conn):

    supp_alerts = pd.DataFrame(columns=['alert_id', 'object_id', 'suppression_reason', 'mute_start_date',
                                            'mute_end_date', 'parent_alert_id', 'recency_threshold',
                                            'similarity_threshold'])
    alerts = pd.DataFrame(columns=['alert_id', 'model_id', 'model_version', 'date_time',
                                   'as_of_date', 'resolution_date', 'object_id', 'object_type',
                                   'probability', 'decision_threshold', 'priority', 'last_updated_by'])

    copy_to_sql(supp_alerts, table_name="suppressed_alerts",
                    if_exists="append", schema_name=db_name)
    copy_to_sql(alerts, table_name="alerts",
                    if_exists="append", schema_name=db_name)

# if acct_no not exists in alert table, add it in alert table.
def alert_not_exists(local_exp, database_name, conn):
    acct_no = local_exp["acct_no"]
    alerts_in_db = pd.read_sql(
        f"select object_id from {database_name}.alerts WHERE object_id in ({acct_no})",
        conn)

    absent_acct_number_rows = local_exp[~alerts_in_db.isin(acct_no)]


    alerts_df = pd.DataFrame(columns=['alert_id', 'model_id', 'model_version', 'date_time',
                                   'as_of_date', 'resolution_date', 'object_id', 'object_type',
                                   'probability', 'decision_threshold', 'priority', 'last_updated_by'])

    alerts_df['object_id'] = absent_acct_number_rows["acct_no"]
    alerts_df["object_type"] = "account"
    alerts_df['date_time'] = int(time.time())
    alerts_df['as_of_date'] = absent_acct_number_rows["as_of_date"]
    alerts_df["alert_id"] = alerts_df["alert_id"].apply(lambda _: uuid.uuid4(), axis=1)
    alerts_df["model_id"] = absent_acct_number_rows["model_id"]
    alerts_df["model_version"] = absent_acct_number_rows["model_version"]
    alerts_df["alert_status"] = "open"
    alerts_df["resolution_date"] = None
    alerts_df["probability"] = absent_acct_number_rows["probability_1"]
    alerts_df["decision_threshold"] = 0.5
    alerts_df["priority"] = "Need to be implemented"
    alerts_df["last_updated_by"] = "system"

    copy_to_sql(alerts_df, table_name="alerts",
                    if_exists="append", schema_name=database_name)

def handle_first_case():

    suppressed_alerts = pd.read_sql(
        f"select * from {database_name}.suppressed_alerts WHERE object_id in ({acct_no}) and suppression_reason = 'end_user_custom' and mute_start_date > {current_time} and mute_end_date < {current_time}",
        conn)

    alerts_df = pd.DataFrame(columns=['alert_id', 'model_id', 'model_version', 'date_time',
                                   'as_of_date', 'resolution_date', 'object_id', 'object_type',
                                   'probability', 'decision_threshold', 'priority', 'last_updated_by'])


#   Need to verify the following data frame. Mistakes present
    alerts_df['object_id'] = suppressed_alerts["acct_no"]
    alerts_df["object_type"] = "account"
    alerts_df['date_time'] = int(time.time())
    alerts_df['as_of_date'] = suppressed_alerts["as_of_date"]
    alerts_df["alert_id"] = alerts_df["alert_id"].apply(lambda _: uuid.uuid4(), axis=1)
    alerts_df["model_id"] = suppressed_alerts["model_id"]
    alerts_df["model_version"] = suppressed_alerts["model_version"]
    alerts_df["alert_status"] = "suppressed"
    alerts_df["resolution_date"] = None
    alerts_df["probability"] = absent_acct_number_rows["probability_1"]
    alerts_df["decision_threshold"] = 0.5
    alerts_df["priority"] = "Need to be implemented"
    alerts_df["last_updated_by"] = "system"

    copy_to_sql(alerts_df, table_name="alerts",
                    if_exists="append", schema_name=database_name)


    #Now add the same row in suppressed alerts table as well.


def alerts_exists(database_name, acct_no, conn):
    handle_first_case()


def generate_alert(local_exp):  # local_exp contains scoring results

    create_tables()

    alert_not_exists(local_exp)

    alerts_exists(local_exp)

