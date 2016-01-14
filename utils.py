from sklearn import preprocessing
import pandas as pd

"""
Library of utility function to tranform DataFrames (columns and values
manipulations)
"""


def transform_dates(data_f):
    """
    Transform the date feature into 3 useful new features: Year, Month, weekday 

    Args:
        data_f (DataFrame): input Pandas DataFrame

    Returns:
        transformed DataFrame
    """
    data_f['Date'] = pd.to_datetime(pd.Series(data_f['Original_Quote_Date']))
    data_f = data_f.drop('Original_Quote_Date', axis=1)
    data_f['Year'] = data_f['Date'].apply(lambda x: int(str(x)[:4]))
    data_f['Month'] = data_f['Date'].apply(lambda x: int(str(x)[5:7]))
    data_f['weekday'] = data_f['Date'].dt.dayofweek
    data_f = data_f.drop('Date', axis=1)
    return data_f


def transform_categorical_features_test_train(data_f_test, data_f_train):
    """
    Transforms categorical features into ints using the LabelEncoder from
    sklearn. Possible values are taken from both given DataFrames (train and test)

    Args:
        data_f (DataFrame): input Pandas DataFrame

    Returns:
        transformed DataFrame
    """
    categorical_columns = [col for col in data_f_test.columns if data_f_test[col].dtype=='object']
    for col in categorical_columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(data_f_test[col].values) + list(data_f_train[col].values))
        data_f_test[col] = lbl.transform(list(data_f_test[col].values))
        data_f_train[col] = lbl.transform(list(data_f_train[col].values))
    return data_f_test, data_f_train


def transform_categorical_features_train(data_f):
    """
    Transforms categorical features into ints using the LabelEncoder from
    sklearn. Useful for cross validation on a train dataset

    Args:
        data_f (DataFrame): input Pandas DataFrame

    Returns:
        transformed DataFrame
    """
    categorical_columns = [col for col in data_f.columns if data_f[col].dtype=='object']
    for col in categorical_columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(data_f[col].values))
        data_f[col] = lbl.transform(list(data_f[col].values))
    return data_f
