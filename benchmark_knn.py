import numpy as np
from sklearn import neighbors, cross_validation
import utils
import file_handler as fh

"""
Script to benchmark the performances of KNN (sklearn library)
Launches cross validations of the model with variation on k and
the number of features to consider (we use the 20 most important
features given by the xgboost training, sorted by importance order)
"""

k_list = [10,25,50,75]

# 20 most important features given by XGBoost (by descending order)
features = ['PropertyField37','SalesField5','PersonalField9','Field7','PersonalField2',
'PersonalField1','SalesField4','PersonalField10A','SalesField1B', 'PersonalField10B',
'PersonalField12','CoverageField9', 'CoverageField11B','PropertyField26B','PropertyField24A',
'PersonalField4B', 'PersonalField15','Field8','PropertyField39B', 'CoverageField11A']

train = fh.load_data('train')

# Data transformation
y = train.QuoteConversion_Flag.values
train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
train = train.fillna(-1)
train = utils.transform_categorical_features_train(train)

# normalization
train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

for index_feat in xrange(1, len(features)):
	tmp_train = train.loc[:, features[:index_feat]]
	for k in k_list:
		clf = neighbors.KNeighborsClassifier(k) 
		scores = cross_validation.cross_val_score(clf, train.values, y, cv=3, scoring='roc_auc')
		print "\nNb of features: %d, K = %d" % (index_feat, k)
		print scores

