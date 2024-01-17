"""
DataRealtor converts a dataset into modelling food.
"""

import os
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE

class DataRealtor:
    """
    This module:
    - Fetches datasets you provide
    - Cleans any unsuitable values
    - Morphs datasets into modelling friendly structure
    """

    def __init__(self, path_to_file, target_feature) -> None:
        self.target = target_feature

        df = pd.read_csv(path_to_file)
        df = self._manage_missing(df)

        # Separating Target and Features
        self.x, self.y = df.pop(target_feature)

        self.modelling_method = None

    def _manage_missing(self, df):
        """
        Removes any row with missing value.
        !! Desired Imputation may be implemented here !!
        """

        return df.dropna(axis=0)
    
    def _manage_outlier(self):
        """
        Tukey's Method to remove outliers.
        This will improve model accuracy. Since purpose of this repository is preliminary evaluation.
        We can afford data loss.
        """

        q1 = self.y.quantile(0.25)
        q3 = self.y.quantile(0.75)

        # Inter Quartile Range
        iqr = q3 - q1

        # Finding the lower and upper bounds for outlier detection
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Find outlier rows on both sides
        is_outlier_lower = (self.y < lower_bound)
        is_outlier_upper = (self.y > upper_bound)

        is_outlier = is_outlier_lower | is_outlier_upper

        self.x = self.x[~is_outlier]
        self.y = self.y[~is_outlier]

    def _manage_imbalance(self):
        """
        SMOTE-ing is the easiest and less controversial way to deal with data imbalance.
        If minority class is less than 20% of the data, SMOTE is applied.
        """
        class_distribution = Counter(self.y)
        minority_class_ratio = class_distribution[min(class_distribution, key=class_distribution.get)] / len(self.y)
        
        if minority_class_ratio < 0.2:
            smote = SMOTE()
            self.x, self.y = smote.fit_resample(self.x, self.y)
