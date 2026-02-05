import model_modules
import json

dataConfig = json.load(open('./scheduler/dataset_template.json'))
modelConfig = json.load(open('config.json'))

model_modules.train(dataConfig, modelConfig, model_version=111, model_id=222 )

