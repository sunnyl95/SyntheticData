"""Metadata module."""

from syntheticdata.metadata import visualization
from syntheticdata.metadata.dataset import Metadata
from syntheticdata.metadata.errors import MetadataError, MetadataNotFittedError
from syntheticdata.metadata.table import Table

__all__ = (
    'Metadata',
    'MetadataError',
    'MetadataNotFittedError',
    'Table',
    'visualization'
)
