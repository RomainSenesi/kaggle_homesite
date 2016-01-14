import pandas as pd
import numpy as np
import xgboost as xgb
import utils
import file_handler as fh

"""
Script to benchmark the performances of XGBoost
Launches cross validations of the model, we can choose some
parameters to make vary in order to compare performances and
tune the parameters values
"""

train = fh.load_data('train')

# Data transformation
y = train.QuoteConversion_Flag.values
train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
train = utils.transform_dates(train)
train = train.fillna(-1)
train = utils.transform_categorical_features_train(train)
data_dm = xgb.DMatrix(train.values, y)  

# base values for parameters
xgb_base_parameters = {  
	'nthread':-1,
	'n_estimators':25,
    'max_depth':15,
    'learning_rate':0.025,
    'silent':True,
    'subsample': 1,
    'colsample_bytree':0.9,
    'objective':'binary:logistic'
}

# lists of values to tune parameters
xgb_tune_parameters = {
	'max_depth': [10, 15, 20],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.],
    'learning_rate': [0.01, 0.025, 0.05],
}

# choose name of the parameter which should vary
chosen_param = 'subsample'  

print "Start cross validation, with variation on: ", chosen_param
for val in xgb_tune_parameters[chosen_param]:
	print "\nValue:", val
	xgb_base_parameters[chosen_param] = val
	print xgb.cv(xgb_base_parameters, data_dm, 10, 
		nfold=2, metrics={'error', 'auc'}, seed = 1200)   
