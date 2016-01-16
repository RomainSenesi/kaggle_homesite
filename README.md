# Kaggle Homesite scripts

Scripts and libraries used to participate to the Kaggle competition "Homesite Quote Conversion", where the objective is to predict which customers will purchase a quoted insurance plan.
https://www.kaggle.com/c/homesite-quote-conversion

Dataset consists in a training set of 261 features for around 250.000 observations.

The predictive model developped here consists in averaging two simple predictive models (Gradient Boosted classification and K Nearest Neighbours classification). Parameters tuning has been performed thanks to the benchmark scripts.

This model gets a score of 0.96144, where the leader reaches a score of 0.97006 (score comuted on a test set with the  area under the ROC curve metric).

#### Files

* _file_handler.py_: library of functions providing an abstraction level on top of the manipulated files (csv, cache, json,...)
* _summary.py_: library of functions for plotting and describing the dataset's features
* _utils.py_: library of functions to manipulate data (dates, categorical features,...)
* _benchmark_knn.py_: benchmark of the Gradient Boosted classification (xgboost library) with parameter tuning
* _benchmark_xgb.py_: benchmark of the K-Nearest-Neighbours classification (sklearn library) with parameter tuning
* _train_models.py_ : script that performs the classifiers training and serializes them into models folder
* _predict.py_ : loads the classifiers from models folder and performs the prediction (output is in results folder)

#### Directory structure

* _data_: contains the dataset in 2 subfolders (originals in _data/csv_, cache in _data/cache_)
* _models_ : contains the classifiers trained and serialized
* _plots_ : directory reserved for plots
* _results_ : contains csv files for Kaggle submission
