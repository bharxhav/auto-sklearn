"""
This module:
    - Fetches all datasets.
    - Morphs datasets into pipeline friendly structures
    - 
"""

import os
import pandas as pd

class DataCleaner:
    """
    Calling class for the library.
    """
    def __init__(self) -> None:
        """
        Catch all datasets in the watch folder
        """
        self.datasets = os.listdir('./datasets/')

    def data_catch_status(self):
        """
        Returns number of datasets available
        """

        true_count = len(self.datasets)

        for filename in self.datasets:
            if 'json' in filename.split('.'):
                true_count -= 1

        return len(self.datasets)

    def get_dataset_pointer(self):
        """
        Not all datasets can be csvs, so this will read all types of datasets.
        This routine will also read data from data_config.json to get even better help!

        Ex: If your data: shades_of_color.txt has a specific structure of delimiters and
            all other pandas parameters, this function will respect it and read it appropriately.
            A file shades_of_color.json must be present to enable this!
        """

        self.handles = []

