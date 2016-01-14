import pandas as pd
import os, os.path as path
import cPickle as pickle
import json

"""
Library providing an abstraction level on the source of the Data/Models/Parameters.

Data:
The two main functions are load_data and save_data which respectively
load and save DataFrames given a basename (e.g load_data('train') will get data from train.csv).
On top of that, these functions are able to handle cache files (using pickle), which
significantly improves the loading time. use_cache and generate_cache options allow to
use this functionality.
Original data is expected in data/csv folder. Generated cache files will be located in
data/cache

Models:

Parameters:
Parameters for the classifiers are stored in json files in the parameters folder. They can be
retrieved as a Python dict with load_parameters (and vice versa with save_parameters).
Example: load_parameters('xgb') will return the parameters for the xgb classifier stored in
parameters/xgb.json
"""


def get_root_dir():
	"""
	Returns:
		String containing the path of the project's root directory
	"""
	return path.dirname(path.abspath(__file__))



def get_csv_path(basename):
	"""
	Returns the path of the csv file containing the data corresponding to
	basename

	Args:
		basename (str): base name of the file
	Returns:
		String containing the path of the csv file
	"""
	return path.join(get_root_dir(), "data", "csv", basename+".csv")



def get_cache_path(basename):
	"""
	Returns the path of the cache file containing the data corresponding to
	basename

	Args:
		basename (str): base name of the file
	Returns:
		String containing the path of the cache file
	"""
	return path.join(get_root_dir(), "data", "cache", basename+".pkl")



def get_parameters_path(basename):
	"""
	Returns the path of the json file containing the parameters for the model
	related to basename

	Args:
		basename (str): base name of the file
	Returns:
		String containing the path of the json file
	"""
	return path.join(get_root_dir(), "parameters", basename+".json")



def get_model_path(basename):
	"""
	Returns the path of the cache file containing the model related to basename

	Args:
		basename (str): base name of the file
	Returns:
		String containing the path of the cache file
	"""
	return path.join(get_root_dir(), "models", basename+".model")



def load_data(tag_name='train', use_cache=True, generate_cache=True):
	"""
	Function to get the data from the basename of the input file. The input file
	can be either a cache file or a csv file (switch with use_cache). If cache option
	is chosen and no cache file is found, data is taken from csv file.
	The cache file can be generated from the csv file using the generate_cache option.

	Args:
		tag_name (str): base name of the csv/cache file to read
		use_cache (bool): indicates if data must be taken from cache file (if exists)
		generate_cache (bool): indicates if a cache file must be generated from the csv
	Returns:
		Pandas DataFrame containing the data
	Raises:
		Exception if no csv file was found when not using the cache
	"""
	cache_path = get_cache_path(tag_name)

	if use_cache and os.path.exists(cache_path):
		data = pickle.load(open(cache_path, 'rb'))
		return data

	# default on csv file reading if cache does not exists (or if use_cache is
	# set to False)
	else:
		csv_path = get_csv_path(tag_name)
		if os.path.exists(csv_path):
			data = pd.read_csv(csv_path)
			if generate_cache:
				generate_cache_file(cache_path, data)
			return data

		# Failed to get the file, raise exception
		else:
			raise Exception("Failure when attempting to get: " + csv_path)



def save_data(data, tag_name='train', generate_csv=True, generate_cache=True):
	"""
	Function that saves the given DataFrame into csv or cache file (depending on
	the options). A protection prevents from overriding original train.csv and test.csv

	Args:
		data (DataFrame): data to serialize
		tag_name (str): base name of the csv/cache file to write
		generate_csv (bool): indicates if a csv file must be generated from the DataFrame
		generate_cache (bool): indicates if a cache file must be generated from the DataFrame
	"""
	if generate_cache:
		cache_path = get_cache_path(tag_name)
		generate_cache_file(cache_path, data)

	if generate_csv:
		if tag_name not in ['train', 'test']:
			csv_path = get_csv_path(tag_name)
			generate_csv_file(csv_path, data)
		else:
			print "Try to overide %s.csv skipped" % tag_name



def generate_csv_file(csv_path, data):
	"""
	Function that writes the input DataFrame into a csv file. It overides the file
	if it already exists

	Args:
		csv_path (str): path of the csv file to create
		data (DataFrame): data to convert into csv
	"""
	print "Generate csv in file: " + csv_path
	if path.exists(csv_path):
		print "Remove current csv..." 
		os.remove(csv_path)
	data.to_csv(csv_path)



def generate_cache_file(cache_path, data):
	"""
	Function that writes the input DataFrame into a cache file. It overides the file
	if it already exists. Data is serialized using cPickle with HIGHEST_PROTOCOL option

	Args:
		cache_path (str): path of the csv file to create
		data (DataFrame): data to convert into cache file
	"""
	print "Generate cache in file: " + cache_path
	if path.exists(cache_path):
		print "Remove current cache..." 
		os.remove(cache_path)
	pickle.dump(data, open(cache_path, 'wb'), pickle.HIGHEST_PROTOCOL)



def save_parameters(model_name, parameters):
	"""
	Saves the parameters of a model in a json file

	Args:
		model_name (str): Name of the model
		parameters (dict): dictionary of the parameters
	"""
	parameters_path = get_parameters_path(model_name)
	print "Generate parameters file: " + parameters_path
	if path.exists(parameters_path):
		print "Remove current cache..." 
		os.remove(parameters_path)
	json.dump(parameters, open(parameters_path, 'w'))



def load_parameters(model_name):
	"""
	Loads the parameters of a model from a json file

	Args:
		model_name (str): Name of the model

	Returns:
		Dictionary of the parameters
	"""
	parameters_path = get_parameters_path(model_name)
	return json.load(open(parameters_path, 'r'))



def save_model(model_name, model):
	"""
	Saves the given model in a cache file

	Args:
		model_name (str): Name of the model
		model (object): Python object containing the model
	"""
	model_path = get_model_path(model_name)
	print "Generate model file: " + model_path
	if path.exists(model_path):
		print "Remove current cache..." 
		os.remove(model_path)
	pickle.dump(model, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)



def load_model(model_name):
	"""
	Loads the given model from a cache file

	Args:
		model_name (str): Name of the model

	Returns:	
		model (object): Python object containing the model
	"""
	model_path = get_model_path(model_name)
	model = pickle.load(open(model_path, 'rb'))
	return model