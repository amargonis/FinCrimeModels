import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle,os,json,uuid
from teradataml.dataframe.copy_to import copy_to_sql
from teradataml.context.context import *
from datetime import datetime
from scipy.spatial import distance_matrix


class finCrimeAlertConfig():
    # Instance level variables
    #  batchCommit = true
    #  featureStoreDb
    #  dataScienceDb
    #  metadataDb
    #  alertThreshold
    #  recencyThreshold
    #  similarityThreshold
    #  alertType        : application defined name or string representing the type of alert - AML, CNP, Bustout, CMS
    
    def __init__(self):
        self.batchCommit = True
        self.featureStoreDb = ""
        self.dataScienceDb = ""
        self.metadataDb = ""
        self.alertThreshold = float(0)
        self.recencyThreshold = int(0)
        self.similarityThreshold = float(0)
        self.alertType = "finCrime"

class finCrimeAlertManager():
    # Global Variables
    alertStatus = {"open":0, "closed": 1}
    resolutionType = {"falsePositive" : 0, "truePositive": 1, "trueNoAction": 2}
    
    
    # Instance level variables
    #  dbConnection   : a Connection to the database
    #  alertConfig    : an instance of the configuration for creating alerts
    def __init__(self, fcAlertConfig, dbConn):
        self.dbConnection = dbConn
        self.alertConfig = fcAlertConfig
        self._columnsList =["alert_id", "alert_status", "alert_type", "resolution_type", "datascience_model_version", 
                      "datascience_model_id", "alert_open_date", "as_of_date", "alert_close_date", 
                      "object_type", "object_id", "alert_score", "decision_threshold", "alert_priority", "alert_description"]

        self._alertsDF = pd.DataFrame(columns=self._columnsList)

    def createAlert(self, scoredObject):
        # prior to creating an alert, determine if an alert already exists
        if self._ifExists(scoredObject):
            # create this as a suppressed alert
            self.suppressAlert(scoredObject)
        else:
            # create this as a new Alert
            #alertObject = {
            #    "alert_id": None,
            #    "alert_status": self.alertStatus['open'],
            #    "alert_type" : self.alertConfig.alertType,
            #    "resolution_type": None,
            #    "datascience_model_version" : scoredObject['datascience_model_version'],
            #    "datascience_model_id" : scoredObject['datascience_model_id'],
            #    "alert_open_date": datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
            #    "as_of_date": scoredObject['as_of_date'],
            #    "alert_close_date": None,
            #    "object_type": scoredObject['object_type'],
            #    "object_id" : int(scoredObject['object_id']),
            #    "alert_score" :  scoredObject['anomaly_score'],
            #    "decision_threshold" : self.alertConfig.alertThreshold,
            #    "alert_priority" : 1,
            #    "alert_description" : None}
            #self._alertsDF = self._alertsDF.append(alertObject, ignore_index=True)
            
            alertObject2 = [
                None,
                self.alertStatus['open'],
                self.alertConfig.alertType,
                None,
                scoredObject['datascience_model_version'],
                scoredObject['datascience_model_id'],
                datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                scoredObject['as_of_date'],
                None,
                scoredObject['object_type'],
                int(scoredObject['object_id']),
                scoredObject['anomaly_score'],
                self.alertConfig.alertThreshold,
                1,
                None]
            self._alertsDF.loc[len(self._alertsDF.index)] = alertObject2
            
            if  not self.alertConfig.batchCommit:
                self.commit()
                
    
    def _ifExists(self, scoredObject): 
        return False

    def commit(self):
        print("===> commit alert batch")
        copy_to_sql(self._alertsDF,
            table_name="alerts", columns_list=self._columnsList,
            if_exists="append", schema_name=self.alertConfig.dataScienceDb)
        # re-initialize the data frame to prevent duplicates
        self._alertsDF = pd.DataFrame(columns=self._columnsList)


#---------------------------------------------- def convert_date(str_date_time):
    #-------------- return datetime.strptime(str_date_time, '%d/%m/%Y_%H:%M:%S')
#------------------------------------------------------------------------------ 
#-------------------------------------------------- def reconvert_date(dateObj):
    #------------------------------ return dateObj.strftime("%d/%m/%Y_%H:%M:%S")
#------------------------------------------------------------------------------ 
# def suppress_custom_alert(data_conf, alert_ads, crime_type, model_version , model_ID, as_of_date, conn):
    #-------------------------------------- ID = data_conf["column_to_preserve"]
    #--------------- str_IDs = ','.join(str(v) for v in alert_ads[ID].to_list())
#------------------------------------------------------------------------------ 
    #----------------------------------------------- alerts_in_db = pd.read_sql(
        # f"""select object_id,alert_id,mute_start_date,mute_end_date from {data_conf["alert_db"]}.alerts_suppressed WHERE mute_start_date is not null and mute_end_date is not null and crime_type = '{crime_type}' and object_id in ({str_IDs})""",
        #----------------------------------------------------------------- conn)
    #--------------- alerts_in_db.object_id = alerts_in_db.object_id.astype(int)
    # alerts_in_db.mute_start_date = alerts_in_db.mute_start_date.apply(convert_date)
    # alerts_in_db.mute_end_date = alerts_in_db.mute_end_date.apply(convert_date)
#------------------------------------------------------------------------------ 
    # alerts_in_db = alerts_in_db[(alerts_in_db["mute_start_date"] <= datetime.now()) & (alerts_in_db["mute_end_date"] >= datetime.now())]
    # alerts_in_db.mute_start_date = alerts_in_db.mute_start_date.apply(reconvert_date)
    # alerts_in_db.mute_end_date = alerts_in_db.mute_end_date.apply(reconvert_date)
#------------------------------------------------------------------------------ 
    # df_all = alert_ads.merge(alerts_in_db.drop_duplicates(), left_on=ID, right_on = 'object_id', how='left', indicator=True)
    #------------- present_acct_number_rows = df_all[df_all['_merge'] == 'both']
    #------------------------------------ del present_acct_number_rows['_merge']
    #--------- absent_acct_number_rows = df_all[df_all['_merge'] == 'left_only']
    #------------------------------------- del absent_acct_number_rows['_merge']
#------------------------------------------------------------------------------ 
    #------------------------------------ if not present_acct_number_rows.empty:
        # alerts_df = pd.DataFrame(columns=['alert_id', 'alert_status', 'crime_type','model_id', 'model_version', 'date_time',
                                       # 'as_of_date', 'resolution_date', 'object_id', 'object_type',
                                       # 'probability', 'decision_threshold', 'priority', 'last_updated_by'])
#------------------------------------------------------------------------------ 
        #----------------- alerts_df['object_id'] = present_acct_number_rows[ID]
        #----------------------------------------- alerts_df["object_type"] = ID
        #---------------------------------- alerts_df["crime_type"] = crime_type
        #- alerts_df['date_time'] = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
        #---------------------------------- alerts_df['as_of_date'] = as_of_date
        # alerts_df["alert_id"] = alerts_df["alert_id"].apply(lambda _: uuid.uuid4().hex)
        #-------------------------------------- alerts_df["model_id"] = model_ID
        #---------------------------- alerts_df["model_version"] = model_version
        #------------------------------ alerts_df["alert_status"] = "suppressed"
        #----------------------------------- alerts_df["resolution_date"] = None
        #---------- alerts_df["probability"] = present_acct_number_rows["Score"]
        #--------------------------------- alerts_df["decision_threshold"] = 0.5
        #------------------------------------------ alerts_df["priority"] = None
        #------------------------------- alerts_df["last_updated_by"] = "system"
#------------------------------------------------------------------------------ 
        #--------------------------- copy_to_sql(alerts_df, table_name="alerts",
                        # if_exists="append", schema_name=data_conf["alert_db"])
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
        # supp_alerts_df = pd.DataFrame(columns=['alert_id','parent_alert_id','object_id','object_type','crime_type','suppression_reason', 'mute_start_date',
                                                # 'mute_end_date', 'recency_value',
                                                #---------- 'similarity_value'])
#------------------------------------------------------------------------------ 
        #-------------------- supp_alerts_df["alert_id"] = alerts_df["alert_id"]
        # supp_alerts_df["parent_alert_id"] = present_acct_number_rows["alert_id"]
        #------------------ supp_alerts_df['object_id'] = alerts_df['object_id']
        #------------------------------------ supp_alerts_df["object_type"] = ID
        #----------------------------- supp_alerts_df["crime_type"] = crime_type
        # supp_alerts_df["mute_start_date"] = present_acct_number_rows["mute_start_date"]
        # supp_alerts_df["mute_end_date"] = present_acct_number_rows["mute_end_date"]
        #-------------- supp_alerts_df["suppression_reason"] = "end_user_custom"
        #-------------------------------- supp_alerts_df["recency_value"] = None
        #----------------------------- supp_alerts_df["similarity_value"] = None
#------------------------------------------------------------------------------ 
        #----------- copy_to_sql(supp_alerts_df, table_name="alerts_suppressed",
                        # if_exists="append", schema_name=data_conf["alert_db"])
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
    # carry_over_custom = absent_acct_number_rows.drop(["alert_id","object_id","mute_start_date","mute_end_date"], axis=1)
    #-------------------------------------------------- return carry_over_custom
#------------------------------------------------------------------------------ 
# def suppress_open_alert(data_conf, carry_over_custom, crime_type, model_version , model_ID, as_of_date, conn):
    #-------------------------------------- ID = data_conf["column_to_preserve"]
    #------- str_IDs = ','.join(str(v) for v in carry_over_custom[ID].to_list())
    #--------------------------------- lst_IDs = carry_over_custom[ID].to_list()
    #----------------------------------------------- alerts_in_db = pd.read_sql(
        # f"""select object_id,alert_id from {data_conf["alert_db"]}.alerts WHERE object_id in ({str_IDs}) and crime_type = '{crime_type}' and alert_status = 'open'""",
        #----------------------------------------------------------------- conn)
#------------------------------------------------------------------------------ 
    #--------------- alerts_in_db.object_id = alerts_in_db.object_id.astype(int)
#------------------------------------------------------------------------------ 
    # df_all = carry_over_custom.merge(alerts_in_db.drop_duplicates(), left_on=ID, right_on = 'object_id', how='left', indicator=True)
    #------------- present_acct_number_rows = df_all[df_all['_merge'] == 'both']
    #--------- absent_acct_number_rows = df_all[df_all['_merge'] == 'left_only']
    #------------------------------------- del absent_acct_number_rows['_merge']
#------------------------------------------------------------------------------ 
    # alerts_df = pd.DataFrame(columns=['alert_id', 'alert_status', 'crime_type', 'model_id', 'model_version', 'date_time',
                                   # 'as_of_date', 'resolution_date', 'object_id', 'object_type',
                                   # 'probability', 'decision_threshold', 'priority', 'last_updated_by'])
#------------------------------------------------------------------------------ 
    #--------------------- alerts_df['object_id'] = present_acct_number_rows[ID]
    #--------------------------------------------- alerts_df["object_type"] = ID
    #-------------------------------------- alerts_df["crime_type"] = crime_type
    #----- alerts_df['date_time'] = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
    #-------------------------------------- alerts_df['as_of_date'] = as_of_date
    # alerts_df["alert_id"] = alerts_df["alert_id"].apply(lambda _: uuid.uuid4().hex)
    #------------------------------------------ alerts_df["model_id"] = model_ID
    #-------------------------------- alerts_df["model_version"] = model_version
    #---------------------------------- alerts_df["alert_status"] = "suppressed"
    #--------------------------------------- alerts_df["resolution_date"] = None
    #-------------- alerts_df["probability"] = present_acct_number_rows["Score"]
    #------------ alerts_df["decision_threshold"] = data_conf["alert_threshold"]
    #---------------------------------------------- alerts_df["priority"] = None
    #----------------------------------- alerts_df["last_updated_by"] = "system"
#------------------------------------------------------------------------------ 
    #------------------------------- copy_to_sql(alerts_df, table_name="alerts",
                    #---- if_exists="append", schema_name=data_conf["alert_db"])
#------------------------------------------------------------------------------ 
    # supp_alerts_df = pd.DataFrame(columns=['alert_id','parent_alert_id','object_id','object_type','crime_type','suppression_reason', 'mute_start_date',
                                            #- 'mute_end_date', 'recency_value',
                                            #-------------- 'similarity_value'])
#------------------------------------------------------------------------------ 
    #------------------------ supp_alerts_df["alert_id"] = alerts_df["alert_id"]
    #-- supp_alerts_df["parent_alert_id"] = present_acct_number_rows["alert_id"]
    #---------------------- supp_alerts_df['object_id'] = alerts_df['object_id']
    #---------------------------------------- supp_alerts_df["object_type"] = ID
    #--------------------------------- supp_alerts_df["crime_type"] = crime_type
    # supp_alerts_df["mute_start_date"] = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
    #------------------------------------ supp_alerts_df["mute_end_date"] = None
    #--------------------- supp_alerts_df["suppression_reason"] = "already open"
    #------------------------------------ supp_alerts_df["recency_value"] = None
    #--------------------------------- supp_alerts_df["similarity_value"] = None
#------------------------------------------------------------------------------ 
    #--------------- copy_to_sql(supp_alerts_df, table_name="alerts_suppressed",
                    #---- if_exists="append", schema_name=data_conf["alert_db"])
#------------------------------------------------------------------------------ 
    # carry_over_open = absent_acct_number_rows.drop(["alert_id","object_id"], axis=1)
    #---------------------------------------------------- return carry_over_open
#------------------------------------------------------------------------------ 
# def open_alert(data_conf, carry_over_closed, crime_type, model_version , model_ID, as_of_date, conn):
    #-------------------------------------- ID = data_conf["column_to_preserve"]
#------------------------------------------------------------------------------ 
    # alerts_df = pd.DataFrame(columns=['alert_id','alert_status','crime_type','model_id', 'model_version', 'date_time',
                               # 'as_of_date', 'resolution_date', 'object_id', 'object_type',
                               # 'probability', 'decision_threshold', 'priority', 'last_updated_by'])
#------------------------------------------------------------------------------ 
    #---------------------------- alerts_df['object_id'] = carry_over_closed[ID]
    #--------------------------------------------- alerts_df["object_type"] = ID
    #-------------------------------------- alerts_df["crime_type"] = crime_type
    #----- alerts_df['date_time'] = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
    #-------------------------------------- alerts_df['as_of_date'] = as_of_date
    # alerts_df["alert_id"] = alerts_df["alert_id"].apply(lambda _: uuid.uuid4().hex)
    #------------------------------------------ alerts_df["model_id"] = model_ID
    #-------------------------------- alerts_df["model_version"] = model_version
    #---------------------------------------- alerts_df["alert_status"] = "open"
    #--------------------------------------- alerts_df["resolution_date"] = None
    #--------------------- alerts_df["probability"] = carry_over_closed["Score"]
    #------------ alerts_df["decision_threshold"] = data_conf["alert_threshold"]
    # alerts_df["priority"] = carry_over_closed["Score"].apply(get_priority_level)
    #----------------------------------- alerts_df["last_updated_by"] = "system"
#------------------------------------------------------------------------------ 
    #------------------------------- copy_to_sql(alerts_df, table_name="alerts",
                    #---- if_exists="append", schema_name=data_conf["alert_db"])
#------------------------------------------------------------------------------ 
# def suppress_closed_alert(data_conf, carry_over_open, crime_type, model_version , model_ID, as_of_date, conn):
    #-------------------------------------- ID = data_conf["column_to_preserve"]
    #-- str_IDs_alerts = ','.join(str(v) for v in carry_over_open[ID].to_list())
    #----------------------------------------------- alerts_in_db = pd.read_sql(
        # f"""select object_id, alert_id, model_version,as_of_date, date_time from {data_conf["alert_db"]}.alerts WHERE object_id in ({str_IDs_alerts}) and crime_type = '{crime_type}' and alert_status = 'closed'""",
            #------------------------------------------------------------- conn)
#------------------------------------------------------------------------------ 
    #--------------- alerts_in_db.object_id = alerts_in_db.object_id.astype(int)
#------------------------------------------------------------------------------ 
    #------------------------------------------------ if not alerts_in_db.empty:
#------------------------------------------------------------------------------ 
        #-------- for group in alerts_in_db.groupby(['date_time','as_of_date']):
            # df_all = carry_over_open.merge(group[1].drop_duplicates(), left_on=ID, right_on = 'object_id', how='left', indicator=True)
            #----- present_acct_number_rows = df_all[df_all['_merge'] == 'both']
            #---------------------------- del present_acct_number_rows['_merge']
            # str_IDs_common = ','.join(str(v) for v in present_acct_number_rows[ID].to_list())
#------------------------------------------------------------------------------ 
            #- absent_acct_number_rows = df_all[df_all['_merge'] == 'left_only']
            #----------------------------- del absent_acct_number_rows['_merge']
#------------------------------------------------------------------------------ 
            #-------------- old_date_time = str(group[1]["date_time"].values[0])
            # time_diff  = datetime.now() - datetime.strptime(old_date_time.split("_")[0], '%d/%m/%Y')
#------------------------------------------------------------------------------ 
            #------ old_model_version = str(group[1]["model_version"].values[0])
            #------------ old_as_of_date = str(group[1]["as_of_date"].values[0])
            # old_score_ADS_name = pd.read_sql(f"""select scoring_table from {data_conf["ADS_db_name"]}.AML_metadata where as_of_date = '{old_as_of_date}' and model_version = '{old_model_version}'""", conn).values[0][0]
            # old_score_ADS = pd.read_sql(f"""select * from {data_conf["ADS_db_name"]}.{old_score_ADS_name} where {ID} in ({str_IDs_common})""", conn)
            # old_as_of_date = str(old_score_ADS[data_conf["Date_col_name"]].values[0])
            # old_score_ADS = old_score_ADS.drop(data_conf["Date_col_name"], axis=1)
            #----------------------- old_score_ADS = old_score_ADS.astype(float)
            #---------------- old_score_ADS = old_score_ADS.sort_values(by=[ID])
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
            # new_score_ADS = pd.read_sql(f"""select * from {data_conf["ADS_db_name"]}.{data_conf["Score_ADS_name"]} where {ID} in ({str_IDs_common})""", conn)
            # new_as_of_date = str(new_score_ADS[data_conf["Date_col_name"]].values[0])
            # new_score_ADS = new_score_ADS.drop(data_conf["Date_col_name"], axis=1)
            #----------------------- new_score_ADS = new_score_ADS.astype(float)
            #---------------- new_score_ADS = new_score_ADS.sort_values(by=[ID])
#------------------------------------------------------------------------------ 
            #--------------------- old_columns = old_score_ADS.columns.to_list()
            #--------------------- new_columns = new_score_ADS.columns.to_list()
#------------------------------------------------------------------------------ 
            #------------- len_common = len(set(old_columns) & set(new_columns))
#------------------------------------------------------------------------------ 
            # similarity_array = 1/(Euclidean_Dist(old_score_ADS, new_score_ADS, new_columns)+1)
#------------------------------------------------------------------------------ 
            #--------- present_acct_number_rows["similarity"] = similarity_array
#------------------------------------------------------------------------------ 
            #---------------------- l = len(present_acct_number_rows.index) // 2
            #---------- present_acct_number_rows.loc[:l - 1, 'similarity'] = 0.5
#------------------------------------------------------------------------------ 
            # present_acct_number_rows_similar = present_acct_number_rows[present_acct_number_rows["similarity"] >= data_conf["similarity_threshold"]]
            # present_acct_number_rows_unsimilar = present_acct_number_rows[present_acct_number_rows["similarity"] <= data_conf["similarity_threshold"]]
#------------------------------------------------------------------------------ 
            # if time_diff.days >= data_conf["recency_threshold"] and not present_acct_number_rows_similar.empty and len_common == max(len(old_columns),len(new_columns)):
                # alerts_df = pd.DataFrame(columns=['alert_id', 'alert_status','crime_type','model_id', 'model_version', 'date_time',
                                               # 'as_of_date', 'resolution_date', 'object_id', 'object_type',
                                               # 'probability', 'decision_threshold', 'priority', 'last_updated_by'])
#------------------------------------------------------------------------------ 
                #- alerts_df['object_id'] = present_acct_number_rows_similar[ID]
                #--------------------------------- alerts_df["object_type"] = ID
                #-------------------------- alerts_df["crime_type"] = crime_type
                # alerts_df['date_time'] = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
                #-------------------------- alerts_df['as_of_date'] = as_of_date
                # alerts_df["alert_id"] = alerts_df["alert_id"].apply(lambda _: uuid.uuid4().hex)
                #------------------------------ alerts_df["model_id"] = model_ID
                #-------------------- alerts_df["model_version"] = model_version
                #---------------------- alerts_df["alert_status"] = "suppressed"
                #--------------------------- alerts_df["resolution_date"] = None
                #-- alerts_df["probability"] = present_acct_number_rows["Score"]
                # alerts_df["decision_threshold"] = data_conf["alert_threshold"]
                #---------------------------------- alerts_df["priority"] = None
                #----------------------- alerts_df["last_updated_by"] = "system"
#------------------------------------------------------------------------------ 
                #------------------- copy_to_sql(alerts_df, table_name="alerts",
                                # if_exists="append", schema_name=data_conf["alert_db"])
#------------------------------------------------------------------------------ 
                # supp_alerts_df = pd.DataFrame(columns=['alert_id','parent_alert_id','object_id','object_type','crime_type','suppression_reason', 'mute_start_date',
                                                        # 'mute_end_date', 'recency_value',
                                                        #-- 'similarity_value'])
#------------------------------------------------------------------------------ 
                #------------ supp_alerts_df["alert_id"] = alerts_df["alert_id"]
                # supp_alerts_df["parent_alert_id"] = present_acct_number_rows_similar["alert_id"]
                #---------- supp_alerts_df['object_id'] = alerts_df['object_id']
                #---------------------------- supp_alerts_df["object_type"] = ID
                #--------------------- supp_alerts_df["crime_type"] = crime_type
                # supp_alerts_df["mute_start_date"] = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
                #------------------------ supp_alerts_df["mute_end_date"] = None
                #--- supp_alerts_df["suppression_reason"] = "similar and recent"
                #------------ supp_alerts_df["recency_value"] = (time_diff.days)
                # supp_alerts_df["similarity_value"] = present_acct_number_rows_similar["similarity"]
#------------------------------------------------------------------------------ 
                #--- copy_to_sql(supp_alerts_df, table_name="alerts_suppressed",
                                # if_exists="append", schema_name=data_conf["alert_db"])
#------------------------------------------------------------------------------ 
                # carry_over_closed = present_acct_number_rows_unsimilar.append(absent_acct_number_rows)
                # open_alert(data_conf, carry_over_closed, crime_type, model_version , model_ID, as_of_date, conn)
            #------------------------------------------------------------- else:
                # carry_over_closed = present_acct_number_rows.append(absent_acct_number_rows)
                # open_alert(data_conf, carry_over_closed, crime_type, model_version , model_ID, as_of_date, conn)
    #--------------------------------------------------------------------- else:
        # open_alert(data_conf, carry_over_open, crime_type, model_version , model_ID, as_of_date, conn)
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 
#------------------------------------------------- def get_priority_level(prob):
    #------------------------------------------------------------ if prob < 0.6:
        #---------------------------------------------------------- return "Low"
    #--------------------------------------------------------- elif prob < 0.75:
        #------------------------------------------------------- return "Medium"
    #--------------------------------------------------------- elif prob < 0.85:
        #-------------------------------------------------- return "Medium-High"
    #--------------------------------------------------------------------- else:
        #--------------------------------------------------------- return "High"
#------------------------------------------------------------------------------ 
#------------------------------------------- def Euclidean_Dist(df1, df2, cols):
    #-------- return np.linalg.norm(df1[cols].values - df2[cols].values, axis=1)
