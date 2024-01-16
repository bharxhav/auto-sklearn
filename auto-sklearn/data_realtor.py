"""
DataRealtor converts a dataset into modelling food.
"""

import os
import pandas as pd

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

    def _manage_missing(self, df):
        """
        Removes any row with missing value.
        !! Desired Imputation may be implemented here !!
        """

        return df.dropna(axis=0)
