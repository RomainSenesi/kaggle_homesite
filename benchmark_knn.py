import pandas as pd
import numpy as np
from sklearn import neighbors, cross_validation
import utils
import file_handler as fh
from time import time
from sklearn.metrics import accuracy_score, roc_auc_score


features = ['PropertyField37','SalesField5','PersonalField9','Field7','PersonalField2',
'PersonalField1','SalesField4','PersonalField10A','SalesField1B', 'PersonalField10B',
'PersonalField12']

# features = ['CoverageField9', 'CoverageField11B','PropertyField26B','PropertyField24A', 'PersonalField4B', 'PersonalField15','Field8','PropertyField39B', 'CoverageField11A', 'PersonalField18', 'PropertyField26A', 'SalesField1A', 'SalesField10', 'Field7', 'SalesField2B', 'PropertyField1A', 'PropertyField39A', 'SalesField6','PersonalField17', 'PersonalField4A', 'PersonalField10A', 'SalesField7','PersonalField10B', 'PersonalField16', 'PropertyField2B','SalesField1B', 'SalesField8']

k_list = [5,10,15,50]

train = fh.load_data('train')

# Data transformation
y = train.QuoteConversion_Flag.values
train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
# train = utils.transform_dates(train)
train = train.fillna(-1)
train = utils.transform_categorical_features_train(train)
train = train.loc[:, features]
# normalization
train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))


clf = neighbors.KNeighborsClassifier(50) 

scores = cross_validation.cross_val_score(clf, train.values, y, cv=3, scoring='roc_auc')
print scores

