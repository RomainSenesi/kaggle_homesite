import pandas as pd
import numpy as np
import xgboost as xgb
import utils
import file_handler as fh

"""
Script that loads the models from models folder, performs the prediction on the test
dataset and output the result in results folder
"""

knn_features = ['PropertyField37','SalesField5','PersonalField9','Field7','PersonalField2',
'PersonalField1','SalesField4','PersonalField10A','SalesField1B', 'PersonalField10B',
'PersonalField12']

if __name__ == "__main__":
	# load data
	train = fh.load_data('train')
	test = fh.load_data('test')

	# transform data 
	Y_train = train.QuoteConversion_Flag.values
	train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
	test = test.drop('QuoteNumber', axis=1)
	train = utils.transform_dates(train)
	test = utils.transform_dates(test)
	train = train.fillna(-1)
	test = test.fillna(-1)
	train, test = utils.transform_categorical_features_test_train(train, test)

	# transform data for knn
	knn_train = train.loc[:, knn_features]
	knn_test = test.loc[:, knn_features]
	knn_train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
	knn_test.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
	print "Data loaded"

	sample = pd.read_csv('results/sample_submission.csv')

	# XGBoost
	clf_xgb = fh.load_model('xgb')
	preds_xgb = clf_xgb.predict_proba(test)[:,1]
	sample.QuoteConversion_Flag = preds_xgb
	sample.to_csv('results/a_xgb_results.csv', index=False)

	# KNN
	clf_knn = fh.load_model('knn')
	preds_knn = clf_knn.predict_proba(knn_test)[:,1]
	sample.QuoteConversion_Flag = preds_knn	
	sample.to_csv('results/a_knn_results.csv', index=False)
	
	# Averages predictions
	preds_avg = (preds_knn + preds_xgb) / 2
	sample.QuoteConversion_Flag = preds_avg	
	sample.to_csv('results/a_avg_results.csv', index=False)
