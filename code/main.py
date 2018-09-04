###############################################################################
import numpy as np
import random as rn

#DO NOT CHANGE THIS
np.random.seed(1478)
rn.seed(2264)
###################

from utils import load_datasets_filenames, load_experiment_configuration
from utils import load_dataset, get_voting_clf

from sklearn.model_selection import StratifiedKFold, train_test_split



if __name__ == "__main__":

	datasets_filenames = load_datasets_filenames()
	config = load_experiment_configuration()

	for dataset_filename in datasets_filenames:
		instances, gold_labels = load_dataset(dataset_filename)
		skfold = StratifiedKFold(n_splits = config["num_folds"],
			                     shuffle = True)
		for train_idxs, test_idxs in skfold.split(X=instances, y=gold_labels):
			possible_train_instances = instances.iloc[train_idxs]
			possible_train_labels = gold_labels.iloc[train_idxs]
			test_instances = instances.iloc[test_idxs].values
			test_labels = instances.iloc[test_idxs].values.ravel()
			
			for sampling_percentage in config["sampling_percentages"]:
				sampled = train_test_split(possible_train_instances,
					                       possible_train_labels,
					                       train_size = sampling_percentage,
				                           stratify = possible_train_labels)

				train_instances = sampled[0].values
				train_gold_labels = sampled[2].values.ravel()

				for strategy in config["generation_strategies"]:
					for base_clf in config["base_classifiers"]:
					    clf_pool = strategy(base_clf(), config["pool_size"])
					    clf_pool.fit(train_instances, train_gold_labels)

					    hard_voting_clf = get_voting_clf(clf_pool)
					    predictions = hard_voting_clf.predict(test_instances)