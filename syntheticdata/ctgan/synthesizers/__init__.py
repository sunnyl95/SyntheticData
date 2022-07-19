"""Synthesizers module."""

from syntheticdata.ctgan.synthesizers.ctgan import CTGANSynthesizer
from syntheticdata.ctgan.synthesizers.tvae import TVAESynthesizer

__all__ = (
    'CTGANSynthesizer',
    'TVAESynthesizer'
)


def get_all_synthesizers():
    return {
        name: globals()[name]
        for name in __all__
    }
