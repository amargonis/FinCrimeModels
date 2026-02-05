import sys
from sklearn import preprocessing

def preprocess(df, overwrite=False, train=False, label_encoders=None, cat_columns_list = []):
    if overwrite == False:
        df_new = df.copy()
    else:
        df_new = df
        
    print (cat_columns_list)
        
    if not cat_columns_list:
        object_feature_mask = df.dtypes==object
        # filter categorical columns using mask and turn it into a list
        object_cols = df.columns[object_feature_mask].tolist()
    
        # Try to cast non-float columns as floats.  If you can't, then keep track as categorical data.
        cat_cols = []
        for object_col in object_cols:
            try:
                df_new[object_col] = df[object_col].astype('float64')
            except: # Categorical columns
                cat_cols.append(object_col)
    else:
        cat_cols = cat_columns_list
          
    if train == True: # Train your own
        # Fit label encoder
        label_encoders = {}

    #object_feature_mask = df.dtypes==object
    # filter categorical columns using mask and turn it into a list
    for cat_col in cat_cols:
        if train == True:
            label_encoders[cat_col] = preprocessing.LabelEncoder().fit(df[cat_col].astype(str).str.strip())
        print ('label_encoder', label_encoders[cat_col].transform(df[cat_col].astype(str).str.strip()))
        df_new[cat_col] = label_encoders[cat_col].transform(df[cat_col].astype(str).str.strip())
        print(cat_col,": ", label_encoders[cat_col].classes_, file=sys.stderr)

    #df_new = df_new.astype(float).fillna(0)
    
    if train == True:
        return df_new, label_encoders
    else:
        return df_new, None