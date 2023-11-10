"""
This module:
    - Fetches all datasets.
    - Morphs datasets into pipeline friendly structures
    - 
"""

import os
import json
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
        self.handles = []

    def data_catch_status(self):
        """
        Returns number of datasets available
        """

        true_count = len(self.datasets)

        for filename in self.datasets:
            if 'json' in filename.split('.'):
                true_count -= 1

        return len(self.datasets)

    def _create_dataset_pointers(self, title, ext):
        """
        Not all datasets can be csvs, so this will read all types of datasets.
        This routine will also read data from data_config.json to get even better help!

        Ex: If your data: shades_of_color.txt has a specific structure of delimiters and
            all other pandas parameters, this function will respect it and read it appropriately.
            A file shades_of_color.json must be present to enable this!
        """

        if ext in 'json':
            return

        params = None
        if f'{title}.json' in self.datasets:
            params = json.load(open(f'{title}.json', 'r'))

        df = None
        loc = f'./{title}.{ext}'

        if ext == 'csv':
            if params:
                df = pd.read_csv(loc, **params)
            else:
                df = pd.read_csv(loc)
        elif ext in ['xls', 'xlsx']:
            if params:
                df = pd.read_excel(loc, **params)
            else:
                df = pd.read_excel(loc)
        elif ext == 'json':
            if params:
                df = pd.read_json(loc, **params)
            else:
                df = pd.read_json(loc)
        elif ext in ['html', 'htm']:
            if params:
                df = pd.read_html(loc, **params)[0]
            else:
                df = pd.read_html(loc)[0]

        self.handles.append(df)

    def curate_dataset_pointers(self):
        """
        Execute this subroutine to read all datasets.
        You will be warned if any of your config is wrong, or if file is not supported.
        """

        for file_name in self.datasets:
            title, ext = file_name.split('.')

            try:
                self._create_dataset_pointers(title, ext)
            except Exception as exc:
                raise(f'LOG:: {title}.{ext} failed reading.' + exc)
