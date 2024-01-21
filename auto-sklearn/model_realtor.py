"""
ModelRealtor trains the following models.

!! NOTE: The order is very important here, as you can limit tasks based on device limitations !!

Regression Models:
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet Regression
5. Polynomial Regression
6. Support Vector Regression (SVR)
7. Decision Tree Regression
8. Random Forest Regression
9. Gradient Boosting Regression
10. AdaBoost Regression
11. Bayesian Ridge Regression
12. Passive Aggressive Regression
13. Huber Regression
14. Theil-Sen Regression
15. LARS Regression
16. RANSAC Regression
17. Gaussian Process Regression
18. Isotonic Regression

Classification Models:
1. Logistic Regression
2. K-Nearest Neighbors (KNeighborsClassifier)
3. Support Vector Machines (SVM) for Classification (SVC)
4. Stochastic Gradient Descent (SGD) Classifier
5. Decision Tree Classifier
6. Random Forest Classifier
7. Gradient Boosting Classifier (GradientBoostingClassifier)
8. AdaBoost Classifier (AdaBoostClassifier)
9. Multilayer Perceptron Classifier (MLPClassifier)
10. Gaussian Naive Bayes (GaussianNB)
11. Bernoulli Naive Bayes (BernoulliNB)
12. Complement Naive Bayes (ComplementNB)
13. Multinomial Naive Bayes (MultinomialNB)
14. Passive Aggressive Classifier
15. Quadratic Discriminant Analysis (QDA)
16. Linear Discriminant Analysis (LDA)
17. Nu-Support Vector Classification
18. One-Class SVM (OneClassSVM)
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,
    PassiveAggressiveRegressor, HuberRegressor, TheilSenRegressor, Lars, RANSACRegressor,
    LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR, SVC, NuSVC, OneClassSVM
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier


class _Model:
    """
    Modular architecture to make modelling abstract.
    """

    def __init__(self, model) -> None:
        self.model = model

    def train(self, x_train, y_train):
        """
        Trains the model.
        """
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        """
        Tests the model.
        """
        return self.model.predict(x_test)

    def metricize_regression(self, predictions, actual):
        """
        Calculate regression metrics: MAE, MSE, RMSE, R2
        """
        metrics = {}

        # Mean Absolute Error
        metrics['MAE'] = mean_absolute_error(actual, predictions)

        # Mean Squared Error
        metrics['MSE'] = mean_squared_error(actual, predictions)

        # Root Mean Squared Error
        metrics['RMSE'] = mean_squared_error(
            actual, predictions, squared=False)

        # R-squared
        metrics['R2'] = r2_score(actual, predictions)

        return metrics

    def metricize_classification(self, predictions, actual):
        """
        Calculate classification metrics: Accuracy, F1, Precision, Recall
        """
        metrics = {}

        # Accuracy
        metrics['Accuracy'] = accuracy_score(actual, predictions)

        # F1-score
        metrics['F1'] = f1_score(
            actual, predictions, average='weighted', zero_division=0)

        # Precision
        metrics['Precision'] = precision_score(
            actual, predictions, average='weighted', zero_division=0)

        # Recall
        metrics['Recall'] = recall_score(
            actual, predictions, average='weighted', zero_division=0)

        return metrics


class ModelRealtor:
    """
    Based on DataRealtor's modelling_method, 18 models are deployed in a distributed architecture!
    This class can also be used independent of DataRealtor.

    Pass either modelling_method or any of {'C', 'R'}.
    - C: Classification Task
    - R: Regression Task
    """

    def __init__(self, data_tuple, task, streams=18) -> None:
        """
        data_tuple is (x_train, x_test, y_train, y_test), each of which are pandas dataframes.
        """
        self.x_train, self.x_test, self.y_train, self.y_test = data_tuple

        self.task = task
        self.results = {}  # To store results

        if self.task == 'C':
            self._deploy_classification_streams(streams)
        elif self.task == 'R':
            self._deploy_regression_streams(streams)
        else:
            raise ValueError(
                "Task must be either 'C' for classification or 'R' for regression.")

    def _deploy_classification_streams(self, stream_limit=18):
        classification_models = [
            ('Logistic Regression', LogisticRegression()),
            ('K-Nearest Neighbors', KNeighborsClassifier()),
            ('Support Vector Machine', SVC()),
            ('Stochastic Gradient Descent', SGDClassifier()),
            ('Decision Tree', DecisionTreeClassifier()),
            ('Random Forest', RandomForestClassifier()),
            ('Gradient Boosting', GradientBoostingClassifier()),
            ('AdaBoost', AdaBoostClassifier()),
            ('Multilayer Perceptron', MLPClassifier()),
            ('Gaussian Naive Bayes', GaussianNB()),
            ('Bernoulli Naive Bayes', BernoulliNB()),
            ('Complement Naive Bayes', ComplementNB()),
            ('Multinomial Naive Bayes', MultinomialNB()),
            ('Passive Aggressive', PassiveAggressiveClassifier()),
            ('Quadratic Discriminant Analysis', QuadraticDiscriminantAnalysis()),
            ('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
            ('Nu-Support Vector Classification', NuSVC()),
            ('One-Class SVM', OneClassSVM())
        ]

        for _, (name, model) in enumerate(classification_models[:stream_limit]):
            print(f"Training {name}...")
            try:
                model_wrapper = _Model(model)
                model_wrapper.train(self.x_train, self.y_train)
                predictions = model_wrapper.predict(self.x_test)
                metrics = model_wrapper.metricize_classification(
                    predictions, self.y_test)
                self.results[name] = metrics
            except Exception as e:
                print(f"Failed to train {name}: {e}")

    def _deploy_regression_streams(self, stream_limit=18):
        regression_models = [
            ('Linear Regression', LinearRegression()),
            ('Ridge Regression', Ridge()),
            ('Lasso Regression', Lasso()),
            ('ElasticNet Regression', ElasticNet()),
            ('Polynomial Regression', make_pipeline(
                PolynomialFeatures(degree=2), LinearRegression())),
            ('Support Vector Regression', SVR()),
            ('Decision Tree Regression', DecisionTreeRegressor()),
            ('Random Forest Regression', RandomForestRegressor()),
            ('Gradient Boosting Regression', GradientBoostingRegressor()),
            ('AdaBoost Regression', AdaBoostRegressor()),
            ('Bayesian Ridge Regression', BayesianRidge()),
            ('Passive Aggressive Regression', PassiveAggressiveRegressor()),
            ('Huber Regression', HuberRegressor()),
            ('Theil-Sen Regression', TheilSenRegressor()),
            ('LARS Regression', Lars()),
            ('RANSAC Regression', RANSACRegressor()),
            ('Gaussian Process Regression', GaussianProcessRegressor()),
            ('Isotonic Regression', IsotonicRegression())
        ]

        for _, (name, model) in enumerate(regression_models[:stream_limit]):
            print(f"Training {name}...")
            try:
                if name == 'Isotonic Regression':
                    # IsotonicRegression requires 1D X
                    x_train = self.x_train.iloc[:, 0].values.reshape(-1, 1)
                    x_test = self.x_test.iloc[:, 0].values.reshape(-1, 1)
                else:
                    x_train, x_test = self.x_train, self.x_test

                model_wrapper = _Model(model)
                model_wrapper.train(x_train, self.y_train)
                predictions = model_wrapper.predict(x_test)
                metrics = model_wrapper.metricize_regression(
                    predictions, self.y_test)
                self.results[name] = metrics
            except Exception as e:
                print(f"Failed to train {name}: {e}")

    def get_best_model(self):
        """
        Returns the name of the best performing model based on R2 score for regression
        or F1 score for classification.
        """
        if not self.results:
            return "No models have been trained yet."

        if self.task == 'R':
            best_model = max(self.results, key=lambda x: self.results[x]['R2'])
        else:  # Classification
            best_model = max(self.results, key=lambda x: self.results[x]['F1'])

        return best_model

    def print_results(self):
        """
        Prints the results for all trained models.
        """
        for model, metrics in self.results.items():
            print(f"\n{model}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        print(f"\nBest performing model: {self.get_best_model()}")
