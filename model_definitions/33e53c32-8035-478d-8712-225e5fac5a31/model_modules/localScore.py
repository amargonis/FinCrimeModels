import scoring
import json

dataConfig = json.load(open('./unitTest/dataset_template.json'))
modelConfig = json.load(open('./unitTest/config.json'))

scoring.score(dataConfig, modelConfig, model_version=111, model_id=222 )
scoring.generateAlerts(dataConfig, modelConfig, model_version=111, model_id=222)

