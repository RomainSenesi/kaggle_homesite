import os.path as path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import file_handler as fh


def get_plot_path(plot_type, column_name):
	"""
	Returns the path of the plot to generate

	Args:
		plot_type (string): type of the plot (e.g boxplots)
		column_name (string): name of the feature for the plot

	Returns:
		$(base directory path)/$(plot_type)/$(column_name).png
	"""
	dir_path = path.dirname(path.abspath(__file__))
	return path.join(dir_path, "plots", plot_type,  column_name+".png")


def generate_boxplots(data_f):
	"""
	Generates the box plots (either count plots or histograms) for each 
	feature of the given DataFrame. The plots are saved in plots/boxplots

	Args:
		data_f (DataFrame): Pandas DataFrame to analyze
	"""
	# plots for numerical columns
	data_num = data_f.select_dtypes(include=["int64"])
	for col in data_num.columns:
		cnt_plt = sns.distplot(data_num.loc[:,col].values)
		cnt_plt.figure.savefig(get_plot_path("boxplots", col))
		plt.clf()

	# plots for categorical columns
	data_cat = data_f.select_dtypes(include=["object"])
	for col in data_cat.columns:
		cnt_plt = sns.countplot(x=col, data=data_cat, palette="Greens_d")
		cnt_plt.figure.savefig(get_plot_path("boxplots", col))
		plt.clf()



def analyze_modes(data_f):
	"""
	Analyze for each feature of the given DataFrame what is the mode
	and count the occurences of the mode

	Args:
		data_f (DataFrame): Pandas DataFrame to analyze

	Returns:
		dict(k,v) with k: column name
					   v: mode of the column and occurences count of the mode
	"""
	result = {}
	data_set_size = len(data_f.index)
	modes = data_f.mode()
	for col in data_f.columns:
		mode = modes.loc[0,col]
		mode_matching_df = data_f[data_f[col] == mode]
		proportion_equal_mode = (1. * len(mode_matching_df.index)) / data_set_size
		result[col] = {"mode": mode, "count": proportion_equal_mode}
	return result



def expand_array(array, size):
	"""
	Adds 0s to the given array to make it have the given size

	Args:
		array (np.array): Numpy array to expand
		size (int): new size of the array

	Returns:
		np.array of the given size based on the given array
	"""
	zeros = np.zeros(size - array.size, dtype=array.dtype)
	return np.concatenate((array, zeros))



def resize_arrays(array_1, array_2):
	"""
	Resize, if needed, the two given arrays so that they have the same size.
	The potential smaller array will be expanded with zeros

	Args:
		array_1 (np.array): first array
		array_2 (np.array): second array

	Returns:
		array_1, array_2 with same size
	"""
	if array_1.size == array_2.size:
		return array_1, array_2
	elif array_1.size < array_2.size:
		return expand_array(array_1, array_2.size), array_2
	else:
		return array_1, expand_array(array_2, array_1.size)



def generate_purity_plots(data_f):
	"""
	Generates the purity plot for each feature of the given DataFrame. Purity plots
	present for each possible value of a given feature, the ratio of QuoteConversion_Flag
	(target feature)
	The plots are saved in plots/purity

	Args:
		data_f (DataFrame): Pandas DataFrame to analyze
	"""
	data_num = data_f.select_dtypes(include=["int64"])	
	for col in data_num.columns:
		# Remove -1 occurences
		df = data_num[data_num[col] != -1]

		# Split on QuoteConversion_Flag value
		df_pos = df[df["QuoteConversion_Flag"] == 1]
		df_neg = df[df["QuoteConversion_Flag"] == 0]

		# Bin counts
		bin_cnt_pos = np.bincount(df_pos[col].values)
		bin_cnt_neg = np.bincount(df_neg[col].values)
		bin_cnt_pos, bin_cnt_neg = resize_arrays(bin_cnt_pos, bin_cnt_neg)
		bin_cnt_tot = bin_cnt_pos + bin_cnt_neg

		# Combine results to have percentages (purity for each possible value)
		pos_ratios = np.true_divide(bin_cnt_pos, bin_cnt_tot)
		neg_ratios = np.true_divide(bin_cnt_neg, bin_cnt_tot)

		# Plot
		f, ax = plt.subplots(1, figsize=(10,5))
		bar_l = [i for i in range(len(bin_cnt_pos))]
		ax.bar(bar_l, pos_ratios, label='Positive', color="#cc99ff", edgecolor='white')
		ax.bar(bar_l, neg_ratios, bottom=pos_ratios, label='Negative', color="#339966", edgecolor='white')
		plt.xticks(np.array(bar_l) + 0.4, bar_l)
		ax.set_ylabel("Percentage")
		plt.legend()
		ax.figure.savefig(get_plot_path("purity", col))
		plt.clf()



if __name__ == "__main__":
	# load data
	data_f = dh.load_data("train")
	print data_f.describe()

	# generate plots (boxplots and purity plots)
	generate_boxplots(data_f)
	generate_purity_plots(data_f)

	# analyze for which features the mode represents more than 99% of the values
	modes =  analyze_modes(data_f)
	filtered_modes = [(col, modes[col]) for col in modes.keys() if modes[col]["count"] < 0.99]
	for f in filtered_modes:
		print f