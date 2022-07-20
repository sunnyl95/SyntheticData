# -*- coding: utf-8 -*-

"""Top-level package for ctgan."""

from syntheticdata.ctgan.synthesizers.ctgan import CTGANSynthesizer
from syntheticdata.ctgan.synthesizers.tvae import TVAESynthesizer

__all__ = (
    'CTGANSynthesizer',
    'TVAESynthesizer',
)
