###############################################################################
from functools import partial

import pandas as pd
from scipy.io import arff

from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

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
	perceptron = partial(Perceptron, max_iter = 100, n_jobs = cpus)
	decision_tree = partial(DecisionTreeClassifier)

	return [perceptron, decision_tree]

def _create_generation_strategies(strategy_percentage, cpus = -1):
	bagging = partial(BaggingClassifier, max_samples = strategy_percentage, 
		              n_jobs = cpus)
	random_subspaces = partial(BaggingClassifier, 
		                       max_features = strategy_percentage,
		                       n_jobs = cpus)

	return [bagging, random_subspaces]

def load_datasets_filenames():
	filenames = ["cm1", "jm1"]
	return filenames

def load_dataset(set_filename):
	SET_PATH = "../data/"
	FILETYPE = ".arff"
	full_filepath = SET_PATH + set_filename + FILETYPE

	data, _ = arff.loadarff(full_filepath)

	dataframe = pd.DataFrame(data)

	gold_labels = pd.DataFrame(dataframe["defects"])
	instances = dataframe.drop(columns = "defects")

	return instances, gold_labels

def get_voting_clf(pool_clf):
	base_estimators = pool_clf.estimators_
	pool_size = len(base_estimators)
	list_estimators = [(str(i), base_estimators[i]) for i in xrange(pool_size)]
	return VotingClassifier(list_estimators, voting = 'hard')