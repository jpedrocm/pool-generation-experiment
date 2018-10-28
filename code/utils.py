###############################################################################
from functools import partial
from math import sqrt
from copy import deepcopy

import json
import pandas as pd
import numpy as np
from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score

from prefit_voting_classifier import PrefitVotingClassifier


def load_experiment_configuration():
	STRATEGY_PERCENTAGE = 0.5

	config = {
	"num_folds": 10,
	"pool_size": 100,
	"strategy_percentage": STRATEGY_PERCENTAGE,
	"sampling_percentages":[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
	"base_classifiers": _create_base_classifiers(),
	"generation_strategies": _create_generation_strategies(STRATEGY_PERCENTAGE)
	}

	return config

def _create_base_classifiers(cpus = -1):
	perceptron = partial(Perceptron, n_jobs = cpus)
	decision_tree = partial(DecisionTreeClassifier)

	return [("Perceptron", perceptron), ("Decision Tree", decision_tree)]

def _create_generation_strategies(strategy_percentage, cpus = -1):
	bagging = partial(BaggingClassifier, max_samples = strategy_percentage, 
		              n_jobs = cpus)
	random_subspaces = partial(BaggingClassifier, 
		                       max_features = strategy_percentage,
		                       n_jobs = cpus)

	return [("Bagging", bagging), ("Random Subspaces", random_subspaces)]

def load_datasets_filenames():
	filenames = ["cm1", "jm1"]
	return filenames

def load_dataset(set_filename):
	SET_PATH = "../data/"
	FILETYPE = ".arff"
	full_filepath = SET_PATH + set_filename + FILETYPE

	data, _ = arff.loadarff(full_filepath)

	dataframe = pd.DataFrame(data)
	dataframe.dropna(inplace=True)

	gold_labels = pd.DataFrame(dataframe["defects"])
	instances = dataframe.drop(columns = "defects")

	return instances, gold_labels

def get_voting_clf(pool_clf):
	base_clfs = pool_clf.estimators_
	clfs_feats = pool_clf.estimators_features_
	pool_size = len(base_clfs)
	clfs_tuples = [(str(i), base_clfs[i]) for i in xrange(pool_size)]
	return PrefitVotingClassifier(clfs_tuples, clfs_feats, voting = 'hard')

def save_predictions(data):
	with open('../predictions/all_predictions.json', 'w') as outfile:
		json.dump(data, outfile)

def load_predictions():
	with open('../predictions/all_predictions.json', 'r') as outfile:
		return json.load(outfile)

def sample_training_data(sampling_percentage, possible_train_instances, 
	                     possible_train_labels):

	if int(sampling_percentage) != 1:
		sample = train_test_split(possible_train_instances,
					              possible_train_labels,
					              train_size = sampling_percentage,
				                  stratify = possible_train_labels)

		return sample[0], sample[2]
	else:
		return possible_train_instances, possible_train_labels

def _g1_score(gold_labels, predicted_labels, average):
	precision = precision_score(gold_labels, predicted_labels, average=average)
	recall = recall_score(gold_labels, predicted_labels, average=average)
	return sqrt(precision*recall)

def _calculate_metrics(gold_labels, predicted_labels):

	metrics = {}
	metrics["auc_roc"] = roc_auc_score(gold_labels, predicted_labels, average='macro')
	metrics["g1"] = _g1_score(gold_labels, predicted_labels, average='macro')
	metrics["f1"] = f1_score(gold_labels, predicted_labels, average='macro')
	metrics["acc"] = accuracy_score(gold_labels, predicted_labels)

	return metrics

def _check_create_dict(given_dict, new_key):
	if new_key not in given_dict.keys():
		given_dict[new_key] = {}

def generate_metrics(predictions_dict):
	metrics = {}

	for set_name, set_dict in predictions_dict.iteritems():
		metrics[set_name] = {}

		for fold, fold_dict in set_dict.iteritems():

			gold_labels = fold_dict["gold_labels"]
			del fold_dict["gold_labels"]

			for sampling_pct, sampling_dict in fold_dict.iteritems():
				_check_create_dict(metrics[set_name], sampling_pct)

				for strategy, strategy_dict in sampling_dict.iteritems():
					_check_create_dict(metrics[set_name][sampling_pct], 
						               strategy)

					for clf, predicted in strategy_dict.iteritems():
						metrics_str = metrics[set_name][sampling_pct][strategy]

						fold_metrics = _calculate_metrics(gold_labels, predicted)

						if clf not in metrics_str.keys():
						    metrics_str[clf] = [fold_metrics]
						else:
							metrics_str[clf].append(fold_metrics)

	return metrics

def _summarize_metrics_folds(metrics_folds):
	summary = {}
	metric_names = metrics_folds[0].keys()

	for metric_name in metric_names:
		scores = [metrics_folds[i][metric_name] for i in xrange(len(metrics_folds))]
		summary[metric_name] = [np.mean(scores), np.std(scores)]

	return summary

def summarize_metrics_folds(metrics_dict):

	summary = deepcopy(metrics_dict)

	for set_name, set_dict in metrics_dict.iteritems():
		for sampling_pct, sampling_dict in set_dict.iteritems():
			for strategy, strategy_dict in sampling_dict.iteritems():
				for clf, metrics_folds in strategy_dict.iteritems():
					cur_summary = _summarize_metrics_folds(metrics_folds)
					summary[set_name][sampling_pct][strategy][clf] = cur_summary

	return summary

def pandanize_summary(summary):

	df = pd.DataFrame(columns = ['set', 'sampling', 'strategy', 'clf',
	                  'mean_auc_roc', 'std_auc_roc', 'mean_acc', 'std_acc',
	                  'mean_f1', 'std_f1', 'mean_g1', 'std_g1'])

	for set_name, set_dict in summary.iteritems():
		for sampling_pct, sampling_dict in set_dict.iteritems():
			for strategy, strategy_dict in sampling_dict.iteritems():
				for clf, summary_folds in strategy_dict.iteritems():
					df_folds = pd.DataFrame(_unfilled_row(4, 8),
						                    columns = df.columns)
					_fill_dataframe_folds(df_folds, summary_folds, set_name,
						                  sampling_pct, strategy, clf)
					df = df.append(df_folds)

	return df.reset_index(drop = True)

def _unfilled_row(str_columns, float_columns):
	row = [" " for i in xrange(str_columns)]
	row.extend([0.0 for j in xrange(float_columns)])
	return [row]

def _fill_dataframe_folds(df, summary, set_name, sampling, strategy, clf):
	df.at[0, "set"] = set_name
	df.at[0, "sampling"] = sampling
	df.at[0, "strategy"] = strategy
	df.at[0, "clf"] = clf
	return _fill_dataframe_metrics(df, summary)

def _fill_dataframe_metrics(df, summary):
	for key, metrics in summary.iteritems():
		df.at[0, "mean_" + key] = metrics[0]
		df.at[0, "std_" + key] = metrics[1]
	return df

def save_pandas_summary(df):
	pd.to_pickle(df, '../metrics/metrics_summary.pkl')

def read_pandas_summary():
	return pd.read_pickle('../metrics/metrics_summary.pkl')

def separate_pandas_summary(df, separate_sets):
	dfs = []

	if separate_sets is True:
		sets = df["set"].unique()
		for set_name in sets:
			dfs.append(df.loc[df["set"]==set_name])
	else:
		dfs.append(df)

	return dfs

def write_comparison(dfs, focus_columns, filename):

	with open('../comparisons/'+ filename + '.txt', "w") as outfile:
		for df_set in dfs:
			if len(dfs) == 1:
				outfile.write("\n\nDATASET: Mixed\n")
			else:
				outfile.write("\n\nDATASET: " + df_set.iat[0,0] + "\n")
			outfile.write("Mean of metrics\n")
			outfile.write(df_set.groupby(by=focus_columns).mean().to_string())
			outfile.write("\n\nStd of metrics\n")
			outfile.write(df_set.groupby(by=focus_columns).std().to_string())
			outfile.write("\n")
			outfile.write("-------------------------------------------------")

def bool_str(s):

    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')

    return s == 'True'