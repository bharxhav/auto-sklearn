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
        self._find_method()

        if self.modelling_method == 'C':
            self._manage_imbalance()
        else:
            self._manage_outlier()

    def _manage_missing(self, df):
        """
        Removes any row with missing value.
        !! Desired Imputation may be implemented here !!
        """

        return df.dropna(axis=0)
    
    def _find_method(self, integer_threshold=10):
        """
        Determines the type of modelling task: Classification or Regression
        based on the data type of the target feature.
        """
        if isinstance(self.y, (int, float)):
            unique_values = sorted(self.y.unique())
            max_value = max(unique_values)
            min_value = min(unique_values)
            
            if max_value - min_value < integer_threshold:
                self.modelling_method = "C"  # Disguised Integer Classification
            else:
                self.modelling_method = "R"  # Regression
        else:
            self.modelling_method = "C"  # Classification
    
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

    def _manage_imbalance(self, minority_threshold=0.2):
        """
        SMOTE-ing is the easiest and less controversial way to deal with data imbalance.
        If minority class is less than 20% of the data, SMOTE is applied.
        """
        class_distribution = Counter(self.y)
        minority_class_ratio = class_distribution[min(class_distribution, key=class_distribution.get)] / len(self.y)
        
        if minority_class_ratio < minority_threshold:
            smote = SMOTE()
            self.x, self.y = smote.fit_resample(self.x, self.y)
    
    def _encode_data(self):
        """
        One-hot encodes categorical features and drops the first column to prevent multicollinearity.
        """
        categorical_columns_x = self.x.select_dtypes(include=['object']).columns
        categorical_columns_y = self.y.select_dtypes(include=['object']).columns
        
        if not categorical_columns_x.empty:
            self.x = pd.get_dummies(self.x, columns=categorical_columns_x, drop_first=True)
        
        if not categorical_columns_y.empty:
            self.y = pd.get_dummies(self.y, drop_first=True)
        
        # Ensure self.y has only one column
        if isinstance(self.y, pd.DataFrame) and len(self.y.columns) > 1:
            self.y = self.y.iloc[:, [0]]

