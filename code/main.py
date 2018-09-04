###############################################################################
import numpy as np
import random as rn

#DO NOT CHANGE THIS
np.random.seed(1478)
rn.seed(2264)
###################

from utils import load_datasets_filenames, load_experiment_configuration
from utils import load_dataset, get_voting_clf, save_predictions

from sklearn.model_selection import StratifiedKFold, train_test_split



if __name__ == "__main__":

	datasets_filenames = load_datasets_filenames()
	config = load_experiment_configuration()
	predictions = {}

	for dataset_filename in datasets_filenames:
		instances, gold_labels = load_dataset(dataset_filename)
		skfold = StratifiedKFold(n_splits = config["num_folds"],
			                     shuffle = True)

		predictions[dataset_filename] = {}

		for fold, division in enumerate(skfold.split(X=instances, y=gold_labels), 1):
			train_idxs = division[0]
			test_idxs = division[1]
			possible_train_instances = instances.iloc[train_idxs]
			possible_train_labels = gold_labels.iloc[train_idxs]
			test_instances = instances.iloc[test_idxs].values
			test_gold_labels = instances.iloc[test_idxs].values.ravel()

			predictions[dataset_filename][fold] = {}
			predictions[dataset_filename][fold]["gold_labels"] = test_gold_labels
			
			for sampling_percentage in config["sampling_percentages"]:
				sampled = train_test_split(possible_train_instances,
					                       possible_train_labels,
					                       train_size = sampling_percentage,
				                           stratify = possible_train_labels)

				train_instances = sampled[0].values
				train_gold_labels = sampled[2].values.ravel()

				preds[dataset_filename][fold][str(sampling_percentage)] = {}
				subpredictions = predictions[dataset_filename][fold][str(sampling_percentage)]

				for strategy_name, strategy in config["generation_strategies"]:
					subpredictions[strategy_name] = {}

					for clf_name, base_clf in config["base_classifiers"]:
					    clf_pool = strategy(base_clf(), config["pool_size"])
					    clf_pool.fit(train_instances, train_gold_labels)

					    hard_voting_clf = get_voting_clf(clf_pool)
					    cur_predictions = hard_voting_clf.predict(test_instances)
					    subpredictions[strategy_name][clf_name] = cur_predictions

	save_predictions(predictions)