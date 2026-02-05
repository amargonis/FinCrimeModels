from teradataml.context.context import * 

conn = None

def create_connection(data_conf):
    global conn
    #print("environ: ")
    #print(os.environ)
    host=os.environ["AOA_CONN_HOST"],
    username = os.environ["AOA_CONN_USERNAME"]
    password = os.environ["AOA_CONN_PASSWORD"]
    logmech = os.getenv("AOA_CONN_LOG_MECH", "LDAP")
    
    #host = data_conf["hostname"]
    #username = os.environ["TD_USERNAME"]
    #password = os.environ["TD_PASSWORD"]
    #logmech = os.getenv("TD_LOGMECH", "TDNEGO")
    print(f"get connection for {host} and {username} using {logmech}")
    #eng = create_context(host=host, username=username, password=password, logmech=logmech)
    eng = create_context(
        host=os.environ["AOA_CONN_HOST"], 
        username=os.environ["AOA_CONN_USERNAME"], 
        password=os.environ["AOA_CONN_PASSWORD"],
        logmech=os.environ["AOA_CONN_LOG_MECH"])
    conn = eng.connect()
    return conn

def close_connection():
    global conn
    conn.close()