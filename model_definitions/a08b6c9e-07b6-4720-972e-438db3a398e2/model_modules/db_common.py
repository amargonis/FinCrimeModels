import os
import pandas as pd
import teradataml as tdml
from teradataml.context.context import *

class db_connection:
    
    def __init__(self, data_conf):
        host = data_conf["hostname"]
        username = os.environ["TD_USERNAME"]
        password = os.environ["TD_PASSWORD"]
        logmech=os.getenv("TD_LOGMECH", "TDNEGO")

        self.eng=create_context(host=host , username=username, password = password, logmech=logmech )
        self.conn = self.eng.connect()
    
    def if_exists_drop_tbl(self, db_name, tbl_name):
        ret=pd.read_sql(f"select distinct TableName FROM DBC.TablesV WHERE DatabaseName= '{db_name}' and TableName='{tbl_name}'", self.conn)
        if len(ret)>0: self.conn.execute(f"DROP TABLE {db_name}.{tbl_name}")
            
    def if_exists(self, db_name, tbl_name):
        ret=pd.read_sql(f"select distinct TableName FROM DBC.TablesV WHERE DatabaseName= '{db_name}' and TableName='{tbl_name}'", self.conn)
        if len(ret)>0: return True
        else: return False
        
    def get_connection (self, data_conf):
        return self.eng, self.conn

