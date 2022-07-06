"""Metrics to evaluate quality of Synthetic Data.

This subpackage exists only to enable importing sdmetrics as part of sdv.
"""

from syntheticdata.metrics import relational, tabular, timeseries

__all__ = [
    'relational',
    'tabular',
    'timeseries',
]
