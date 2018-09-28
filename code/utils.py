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

from prefit_voting_classifier import VotingClassifier


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
	return VotingClassifier(clfs_tuples, clfs_feats, voting = 'hard')

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

def _g_mean_score(gold_labels, predicted_labels, average):
	precision = precision_score(gold_labels, predicted_labels, average=average)
	recall = recall_score(gold_labels, predicted_labels, average=average)
	return sqrt(precision*recall)

def _calculate_metrics(gold_labels, predicted_labels):

	metrics = {}
	metrics["auc"] = roc_auc_score(gold_labels, predicted_labels, average='macro')
	metrics["g_mean"] = _g_mean_score(gold_labels, predicted_labels, average='macro')
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
	                  'auc', 'acc', 'f1', 'g_mean'])

	for set_name, set_dict in summary.iteritems():
		for sampling_pct, sampling_dict in set_dict.iteritems():
			for strategy, strategy_dict in sampling_dict.iteritems():
				for clf, summary_folds in strategy_dict.iteritems():
					summary_folds["clf"] = clf
					summary_folds["strategy"] = strategy
					summary_folds["sampling"] = sampling_pct
					summary_folds["set"] = set_name
					df = df.append(summary_folds, ignore_index = True)

	return df

def save_pandas_summary(pandas_summary):
	pd.to_pickle(pandas_summary, '../results/results_summary.pkl')