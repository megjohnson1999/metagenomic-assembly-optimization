"""Metagenomic Assembly Optimization Package

A comprehensive toolkit for determining optimal assembly strategies
for metagenomic datasets through sample grouping analysis.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.kmer_distance import KmerDistanceCalculator
from .core.metadata_analyzer import MetadataAnalyzer
from .core.grouping_optimizer import GroupingOptimizer
from .core.normalization import KmerNormalizer
from .core.similarity import BiasAwareSimilarity
from .core.batch_correction import BatchCorrector
from .core.bias_assessment import BiasAssessment
from .visualization.visualizer import SampleGroupingVisualizer

__all__ = [
    "KmerDistanceCalculator",
    "MetadataAnalyzer",
    "GroupingOptimizer",
    "KmerNormalizer",
    "BiasAwareSimilarity", 
    "BatchCorrector",
    "BiasAssessment",
    "SampleGroupingVisualizer"
]