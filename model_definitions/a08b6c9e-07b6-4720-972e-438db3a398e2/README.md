# Sample Model 

This repository contains an implementation of MLE based Model for Anti Money Laundering. DecisionForest is trained and evaluation is performed using Forest_Predict and ROC functions.

In order to integrate this model with AnalyticOps, train function has been implemented in training.py which performs training and saves a trained model to Vantage. Similarly evaluate function has been implemented in scoring.py which performs testing/evaluation and save results to models/evaluation.json.

AOA integration also requires three configuration files.
1. config.json where we can provide model hyperparameters and storage information for the model
2. model.json has information about the model. This gets created when a new model is added using "aoa add"
3. data_config.json can be used to create a dataset in AOA

## Integrate Model using AOA UI
In order to integrate and use this model using AOA UI, follow below steps

1. From AOA UI, go to the Projects tab (top left corner of window)
2. Under the Projects, click on CREATE PROJECT button (top right)
3. Fillout information in the Create Project dialog and save it
    - Provide a Name and Description
    - For the Group field add "DEMO"
    - For the Repository provide FincrimeModels Github URL (https://consulting-github.teradata.com/FinancialIndustrySolutions/FinCrimeModels.git)
4. Now you should be able to see FinCrimeModels under Projects

## Dataset Creation
In order to train and test a model you will first need to create a dataset

1. Go to the Projects tab
2. Double click on the FincrimeModels project
2. Inside the project, go to the Datasets tab and click on CREATE DATASET
3. Fill all the fields in "CREATE A NEW DATA SET" and save it
    - For Metadata you can use content from data_config.json (see [here](https://consulting-github.teradata.com/FinancialIndustrySolutions/FinCrimeModels/blob/master/data_config.json))


## Training
Once project and dataset are created, now training can be launched

1. Go to the Projects tab.
2. Double click the FincrimeModels project
3. Here you should be able to see multiple models
4. Double click on "AML MLE-Based Supervised Model"
5. Click the Train Model button
6. From the MODEL TRAINING window, Select the newly created dataset
7. From SELECT A TRAINER, Select the Native Docker Runner
8. Once the Trainer is selected, click the TRAIN MODEL button to launch the training
9. See the logs on UI to verify the training progress

## Evaluation
Once the training gets completed. You can now launch evaluation

1. Click the Evaluate Model button from the same MODEL TRAINING window
2. Select the dataset from SELECT A DATASET dialog
3. Select Native Docker Runner as a Trainer from SELECT A TRAINER dialog
4. Once Trainer is selected, click on EVALUATE MODEL button
5. In order to see evaluation results, go to Models tab
6. Double click on AML MLE-Based Supervised Model
7. Select the model. Here model status should be updated to "Evaluated"
8. Once the model is selected, click the arrow under View Events option (right side)
9. It will open Event Id list. Select the event with Type "Evaluated"
10. Click the arrow under Details option and that will show MODEL SCORING window
11. Here you can Approve or Reject the model

## Scoring/Deployment
Once the evaluation and approvel gets completed. You can now Deploy the model

1. Select the model under Models tab and click on Deploy Model button
2. On the newly opened dialog select "Batch" from the Engine Type option
3. Provide a schedule and deploy the model
4. Open Airflow (currently it can be accessed at http://aoa-url:9090/admin/)
5. After few seconds new model should be available on Airflow dashboard
6. Currently AoA requires dataset_template.json under /tmp/scheduler/ folder, if    dataset_template.json is not there, update and copy the file manually from the scheduler folder of the project
7. Once the model is there on Airflow UI, toggle the status of the deployment to "On" (Click on toggle button available on left side of the DAG id)
8. Trigger the job from the Links section available on the right
9. You can also view the logs from the Links section or change the views e.g. you can select the Graph View to see the detailed job status