# -*- coding: utf-8 -*-
# configure logging for the library with a null handler (nothing is printed by default). See
# http://docs.python-guide.org/en/latest/writing/logging/

"""Top-level package for SDV."""

__author__ = """MIT Data To AI Lab"""
__email__ = 'dailabmit@gmail.com'
__version__ = '0.15.1.dev0'

from syntheticdata import tabular
from syntheticdata.demo import load_tabular_demo

__all__ = (
    'tabular',
    'load_tabular_demo'
)
