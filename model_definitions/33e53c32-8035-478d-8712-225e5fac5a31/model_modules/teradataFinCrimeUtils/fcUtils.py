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
    
    print(f"get connection for {host} and {username} using {logmech}")
    #eng = create_context(host=host, username=username, password=password, logmech=logmech)
    eng = create_context(
        host=os.environ["AOA_CONN_HOST"], 
        username=os.environ["AOA_CONN_USERNAME"], 
        password=os.environ["AOA_CONN_PASSWORD"],
        logmech=os.environ["AOA_CONN_LOG_MECH"], "LDAP")
    conn = eng.connect()
    return conn

def close_connection():
    global conn
    conn.close()