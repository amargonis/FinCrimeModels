# Clustering and Anomaly Detection Model


The application relies upon the following components:
 - Teradata Feature Calculator
 - Feature Calculator Extensions for Data Science

The application is designed to be dynamic and provide explainability and alert scoring.  As such, there is additional complexity in the code to manage these processes.  
    
Main Components:
    training.py train()
    scoring.py score()
    localTest.py sample to call train.
    
The feature calculator integration can be replaced with a custom implementation for feature and model configuration data.

Key data required:
## Feature Values 
v_modelDefinition_{id}_{version}: The view that contains the actual data that will be used as part of the process.

    Create View HCLS.v_modelDefinition_1_1 as 
    Select agg1.as_of_date as fc_agg_summary_date,
    ZEROIFNULL(avg01_provider_activit_all_365) as avg01_provider_activit_all_365,
    ... <other fields>,
    ZEROIFNULL(sum01_provider_activit_all_365) as sum01_provider_activit_all_365,
    HCLS.v_claim_detail.provider_name as v_claim_detail_provider_name 
    FROM <source tables>);

In the above sample:
    fc_agg_summary_date is the observation date of the feature.  Meaning the date that was used to calculate the feature values not the current date or the date features calculation was run, but the "as of" date of the feature values.  Calculate feature values as of "2025-12-31" would calculate year end values and then the model could be run for the year 2025.

    v_claim_detail_provider_name is the unique key of the object to be scored.  Uniqueness is determined by the object key plus the feature fc_agg_summary_date.  The same provider could have an entry for 2024-12-31 and 2025-12-31.

## Feature Set Definition
# See def getClusteredFeatures in featureCalculator.py for examples
v_model_feature: This view contains the list of featues for the model as well as the usage of each feature - categorical, numerical, scoring target, object identifier, etc.
    
    create view v_model_feature
    as
    select model_version, model_id, column_name, 
    feature, is_cluster, is_anomaly, ml_type, ds_type,
    anomaly_pos_weight,anomaly_neg_weight
    from model_feature 

is_cluster (0/1): Use this feature to run kMeans clustering.  is_anomaly (0/1): Use this featue to run Anomaly scoring
pos/neg_weight: How heavily should the feaure be scored.  Users may determine that a negative value is not interesting and a neg_weight could be 0 to indicate not to score a negative value whereas a positive value may be of extra importance and should could as a 2 or 3.  
    
As a simple test, set neg_weight to 0 and pos_weight to 1 for all features.  This will make positive values equally weighted and ignore negative values.
    



