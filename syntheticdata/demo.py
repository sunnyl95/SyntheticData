"""Functions to load demo datasets."""

import logging
import pandas as pd

LOGGER = logging.getLogger(__name__)

def load_tabular_demo(dataset_name=None):
    if dataset_name is None:
        tables = pd.read_csv("../example/data/adult.csv")
        return  tables
