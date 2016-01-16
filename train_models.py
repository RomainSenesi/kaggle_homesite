import numpy as np
import utils
import file_handler as fh
from sklearn import neighbors
import xgboost as xgb

"""
Script to train the different classifiers based on the parameters
tuning done thanks to the benchmark scripts
"""

def train_xgb(X_train, y):
	"""
	Train the XGBoost classifier with the parameters defined
	with benchmark_xgb.py

	Args:
		X_train (pd.DataFrame) : training set
		y (np.array) : target values
	"""  
	clf = xgb.XGBClassifier(n_estimators=25,
                        	nthread=-1,
                        	max_depth=15,
                        	learning_rate=0.025,
                        	silent=False,
                        	subsample=1,
                        	colsample_bytree=0.9)             
	xgb_model = clf.fit(X_train, y, eval_metric="auc")
	fh.save_model('xgb', clf)


def train_knn(X_train, y):
	"""
	Train the KNN classifier with the parameters defined
	with benchmark_knn.py

	Args:
		X_train (pd.DataFrame) : training set
		y (np.array) : target values
	"""
	features = ['PropertyField37','SalesField5','PersonalField9','Field7','PersonalField2',
	'PersonalField1','SalesField4','PersonalField10A','SalesField1B', 'PersonalField10B',
	'PersonalField12']
	train = X_train.loc[:, features]
	train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
	clf = neighbors.KNeighborsClassifier(50)
	clf.fit(train.values, y)
	res = clf.predict_proba(train.values)[:,1]
	fh.save_model('knn', clf)



if __name__ == "__main__":
	# load data
	train = fh.load_data('train')
	test = fh.load_data('test')

	# transform data
	y = train.QuoteConversion_Flag.values
	train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
	train = utils.transform_dates(train)
	train = train.fillna(-1)
	train, test = utils.transform_categorical_features_test_train(train, test)

	# train classifiers
	train_xgb(train, y)
	train_knn(train, y)
