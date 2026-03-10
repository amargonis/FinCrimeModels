import training
import json

dataConfig = json.load(open('./unitTest/dataset_template.json'))
modelConfig = json.load(open('./unitTest/config.json'))

training.train(dataConfig, modelConfig, model_version=111, model_id=222 )

