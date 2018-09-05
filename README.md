# Homework 1

This is the first homework for the Multiple Classifiers System's class.

## Description

The goal of this homework is to perform an experiment comparing two different pool generation strategies for ensemble of classifiers using two binary classification datasets. Bagging and Random Subspace were the chosen strategies, which were applied to generate a pool of a 100 base-classifiers each. This is done for both Perceptron and Decision Trees using 10-fold cross-validation. Metrics are collected for each fold and for the whole experiment in the form of means and standard deviations. They include accuracy, f-measure, AUC and g-mean.

## Getting Started

### Requirements
```
* [Python](https://www.python.org/) >= 2.7.15
* [NumPy](http://www.numpy.org/) >= 1.15.1
* [SciPy](https://www.scipy.org/) >= 1.1.0
* [pandas](https://pandas.pydata.org/) >= 0.23.4
* [scikit-learn](http://scikit-learn.org/stable/) >= 0.19.1
```

### Installing

* Clone this repository into your machine
* Download and install all the requirements listed above in the given order
* Download the CM1 and JM1 software defect prediction datasets in .arff format from the [Promise repository](http://promise.site.uottawa.ca/SERepository/datasets-page.html) and do not change their names
* Place both .arff files inside the data/ folder

### Reproducing

* Enter into the code/ folder in your local repository
* Run the experiment to produce every ensemble's predictions
```
python main.py
```
* Once finished, generate all results
```
python generate_results.py
```

## Project Structure

    .            
    ├── code                             # Code files
    │   ├── generate_results.py          # generate metric results
    │   ├── main.py                      # generate models predictions
    │   ├── prefit_voting_classifier.py  # voting classifier for prefit base classifiers
    │   └── utils.py                     # utils functions
    ├── data                             # Datasets files
    ├── predictions                      # Models predictions files
    ├── results                          # Metrics files
    ├── LICENSE.md
    └── README.md

## Author

* [jpedrocm](https://github.com/jpedrocm)

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgements

Thanks to @tomquisel for providing [here](https://gist.github.com/tomquisel/a421235422fdf6b51ec2ccc5e3dee1b4) an initial version of a Voting Classifier for prefit base classifiers.