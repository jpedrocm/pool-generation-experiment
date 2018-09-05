###############################################################################
from functools import partial
from math import sqrt

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

def _g_mean_score(gold_labels, predicted_labels):
	precision = precision_score(gold_labels, predicted_labels)
	recall = recall_score(gold_labels, predicted_labels)
	return sqrt(precision*recall)

def _calculate_metrics(gold_labels, predicted_labels):

	metrics = {}
	#metrics["auc"] = roc_auc_score(gold_labels, predicted_labels)
	metrics["g_mean"] = _g_mean_score(gold_labels, predicted_labels)
	metrics["f1"] = f1_score(gold_labels, predicted_labels)
	metrics["acc"] = accuracy_score(gold_labels, predicted_labels)

	return metrics

def generate_metrics(predictions_dict):
	metrics = {}

	for set_name, set_dict in predictions_dict.iteritems():
		metrics[set_name] = {}

		for fold, fold_dict in set_dict.iteritems():

			gold_labels = fold_dict["gold_labels"]
			del fold_dict["gold_labels"]

			for sampling_pct, sampling_dict in fold_dict.iteritems():
				metrics[set_name][sampling_pct] = {}

				for strategy, strategy_dict in sampling_dict.iteritems():
					metrics[set_name][sampling_pct][strategy] = {}

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
	metrics_names = metrics_folds[0].keys()

	for metric_name in metrics_names:
		scores = [score for score in metrics_folds[i][metric_name] for i in xrange(len(metrics_folds))]
		summary[metric_name] = [np.mean(scores), np.std(scores)]

	return summary

def summarize_metrics_folds(metrics_dict):

	summary = metrics_dict.deepcopy()

	for set_name, set_dict in metrics_dict.iteritems():
		for sampling_pct, sampling_dict in set_dict.iteritems():
			for strategy, strategy_dict in sampling_dict.iteritems():
				for clf, metrics_all_folds in strategy_dict.iteritems():
					this_summary = _summarize_metrics_folds(metrics_all_folds)
					summary[set_name][sampling_pct][strategy][clf] = this_summary

	return summary

def summarize_metrics_pool(metrics_dict):
	pool_summary = {}

	for set_name, set_dict in metrics_dict.iteritems():
		for sampling_pct, sampling_dict in set_dict.iteritems():
			for strategy, strategy_dict in sampling_dict.iteritems():
				if strategy not in pool_summary.keys():
					pool_summary[strategy] = []
				for clf, metrics_summary in strategy_dict.iteritems():
					pool_summary[strategy].append(metrics_summary)

	return _calculate_summary(pool_summary)

def _calculate_summary(point_summary):
	summary = {}
	for key, metrics_list in point_summary.iteritems():
		summary[key] = _calculate_key_summary(metrics_list)

	return summary

def _calculate_key_summary(metrics_list_dict):
	key_summary = {}
	metric_names = key_summary[0].keys()

	for metric_name in metric_names:
		means = [score_tuple[0] in metrics_list_dict[i][metric_name] for i in xrange(len(metrics_list_dict))]
		stds = [score_tuple[1] in metrics_list_dict[i][metric_name] for i in xrange(len(metrics_list_dict))]
		mean_of_means = np.mean(means)
		std_of_means = np.std(means)
		mean_of_stds = np.mean(stds)
		std_of_stds = np.std(stds)
		key_summary[metric_name] = [mean_of_means, std_of_means, mean_of_stds, std_of_stds]

	return key_summary


def save_metrics_summary(metrics_summary, filename):
	with open('../results/' + filename + '.txt', 'w') as outfile:
		for key in metrics_summary:
			print key
			print metrics_summary[key]