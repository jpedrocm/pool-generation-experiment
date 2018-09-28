# Homework 1

![python](https://img.shields.io/badge/python-2.7-blue.svg)
![status](https://img.shields.io/badge/status-finished-green.svg)
![license](https://img.shields.io/badge/license-MIT-green.svg)

This is the first homework for the Multiple Classifiers System's class.

## Description

The goal of this homework is to perform an experiment comparing two different pool generation strategies for ensemble of classifiers using two binary classification datasets. Bagging and Random Subspace were the chosen strategies, which were applied to generate a pool of a 100 base-classifiers each. This is done for both Perceptron and Decision Trees using 10-fold cross-validation. Metrics are collected for each fold and for the whole experiment in the form of means and standard deviations. They include accuracy, f-measure, AUC and g-mean.

## Getting Started

### Requirements

* [Python](https://www.python.org/) >= 2.7.15
* [NumPy](http://www.numpy.org/) >= 1.15.1
* [SciPy](https://www.scipy.org/) >= 1.1.0
* [pandas](https://pandas.pydata.org/) >= 0.23.4
* [scikit-learn](http://scikit-learn.org/stable/) >= 0.19.1


### Installing

* Clone this repository into your machine
* Download and install all the requirements listed above in the given order
* Download the CM1 and JM1 software defect prediction datasets in .arff format from the [Promise repository](http://promise.site.uottawa.ca/SERepository/datasets-page.html) and do not change their names
* Place both .arff files inside the data/ folder

### Reproducing

* Enter into the code/ folder in your local repository
* Run the experiment to produce every ensemble's predictions
```
python generate_predictions.py
```
* Generate all metric results
```
python generate_metrics.py
```
* Then, compare the scenarios wanted
```
python compare_scenarios.py [-f FILENAME] [-s SEPARATE] [-c1 COLUMN1] [-c2 COLUMN2]
```

## Project Structure

    .            
    ├── code                             # Code files
    |   ├── compare_scenarios.py         # Compare metric results 
    │   ├── generate_results.py          # Generate metric results
    │   ├── generate_predictions.py      # Generate models predictions
    │   ├── prefit_voting_classifier.py  # Voting classifier for prefit base classifiers
    │   └── utils.py                     # Utils functions
    ├── comparisons                      # Result comparison files
    ├── data                             # Datasets files
    ├── metrics                          # Metrics files
    ├── predictions                      # Models predictions files
    ├── LICENSE.md
    └── README.md

## Author

* [jpedrocm](https://github.com/jpedrocm)

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgements

Thanks to @tomquisel for providing [here](https://gist.github.com/tomquisel/a421235422fdf6b51ec2ccc5e3dee1b4) an initial version of a Voting Classifier for prefit base classifiers.