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

import pandas as pd

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

        if self.task == 'C':
            self._deploy_classification_streams(streams)
        elif self.task == 'R':
            self._deploy_regression_streams(streams)

    def _deploy_classification_streams(self, stream_limit=18):
        pass

    def _deploy_regression_streams(self, stream_limit=18):
        pass
