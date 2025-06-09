"""Unit tests for k-mer normalization methods."""

import unittest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from core.normalization import KmerNormalizer


class TestKmerNormalizer(unittest.TestCase):
    """Test cases for KmerNormalizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.normalizer = KmerNormalizer()
        
        # Create test count matrix (samples x features)
        np.random.seed(42)
        self.n_samples = 6
        self.n_features = 20
        
        # Create realistic count data with different library sizes
        base_counts = np.random.poisson(10, (self.n_samples, self.n_features))
        
        # Simulate different library sizes
        library_sizes = [1000, 2000, 1500, 3000, 1200, 2500]
        self.count_matrix = np.zeros_like(base_counts, dtype=float)
        
        for i in range(self.n_samples):
            total_counts = np.sum(base_counts[i, :])
            if total_counts > 0:
                self.count_matrix[i, :] = base_counts[i, :] * library_sizes[i] / total_counts
        
        # Create DataFrame version
        self.count_df = pd.DataFrame(
            self.count_matrix,
            index=[f"Sample_{i}" for i in range(self.n_samples)],
            columns=[f"Feature_{i}" for i in range(self.n_features)]
        )
    
    def test_tss_normalization(self):
        """Test Total Sum Scaling normalization."""
        normalized = self.normalizer.normalize(self.count_matrix, method='tss')
        
        # Check that each sample sums to 1 (relative abundance)
        sample_sums = np.sum(normalized, axis=1)
        np.testing.assert_array_almost_equal(sample_sums, np.ones(self.n_samples))
        
        # Check normalization factors are total counts
        expected_factors = np.sum(self.count_matrix, axis=1)
        np.testing.assert_array_almost_equal(
            self.normalizer.normalization_factors, expected_factors
        )
    
    def test_css_normalization(self):
        """Test Cumulative Sum Scaling normalization."""
        normalized = self.normalizer.normalize(self.count_matrix, method='css')
        
        # Check that output has same shape
        self.assertEqual(normalized.shape, self.count_matrix.shape)
        
        # Check that normalization factors are reasonable
        self.assertIsNotNone(self.normalizer.normalization_factors)
        self.assertEqual(len(self.normalizer.normalization_factors), self.n_samples)
        
        # Check that factors are positive
        self.assertTrue(np.all(self.normalizer.normalization_factors > 0))
    
    def test_rle_normalization(self):
        """Test Relative Log Expression normalization."""
        normalized = self.normalizer.normalize(self.count_matrix, method='rle')
        
        # Check that output has same shape
        self.assertEqual(normalized.shape, self.count_matrix.shape)
        
        # Check that size factors have geometric mean close to 1
        log_factors = np.log(self.normalizer.normalization_factors)
        geometric_mean = np.exp(np.mean(log_factors))
        self.assertAlmostEqual(geometric_mean, 1.0, places=3)
    
    def test_tmm_normalization(self):
        """Test Trimmed Mean of M-values normalization."""
        normalized = self.normalizer.normalize(self.count_matrix, method='tmm')
        
        # Check that output has same shape
        self.assertEqual(normalized.shape, self.count_matrix.shape)
        
        # Check that normalization factors have geometric mean close to 1
        log_factors = np.log(self.normalizer.normalization_factors)
        geometric_mean = np.exp(np.mean(log_factors))
        self.assertAlmostEqual(geometric_mean, 1.0, places=3)
    
    def test_clr_normalization(self):
        """Test Centered Log-Ratio transformation."""
        normalized = self.normalizer.normalize(self.count_matrix, method='clr')
        
        # Check that output has same shape
        self.assertEqual(normalized.shape, self.count_matrix.shape)
        
        # Check that each sample has mean log-ratio of 0
        sample_means = np.mean(normalized, axis=1)
        np.testing.assert_array_almost_equal(sample_means, np.zeros(self.n_samples))
    
    def test_dataframe_input_output(self):
        """Test that DataFrame input returns DataFrame output."""
        normalized = self.normalizer.normalize(self.count_df, method='tss')
        
        # Check that output is DataFrame
        self.assertIsInstance(normalized, pd.DataFrame)
        
        # Check that index and columns are preserved
        self.assertEqual(list(normalized.index), list(self.count_df.index))
        self.assertEqual(list(normalized.columns), list(self.count_df.columns))
    
    def test_evaluate_normalization(self):
        """Test normalization evaluation."""
        normalized = self.normalizer.normalize(self.count_matrix, method='css')
        
        metrics = self.normalizer.evaluate_normalization(
            self.count_matrix, normalized
        )
        
        # Check that metrics are returned
        self.assertIn('cv_reduction', metrics)
        self.assertIn('cv_reduction_pct', metrics)
        self.assertIn('raw_dispersion', metrics)
        self.assertIn('normalized_dispersion', metrics)
        
        # CV reduction should be a number
        self.assertIsInstance(metrics['cv_reduction'], (int, float))
        self.assertIsInstance(metrics['cv_reduction_pct'], (int, float))
    
    def test_invalid_method(self):
        """Test that invalid normalization method raises error."""
        with self.assertRaises(ValueError):
            self.normalizer.normalize(self.count_matrix, method='invalid_method')
    
    def test_empty_input(self):
        """Test handling of empty input."""
        empty_matrix = np.zeros((3, 5))
        
        # TSS should handle zeros gracefully
        normalized = self.normalizer.normalize(empty_matrix, method='tss')
        self.assertEqual(normalized.shape, empty_matrix.shape)
    
    def test_normalization_reproducibility(self):
        """Test that normalization is reproducible."""
        norm1 = self.normalizer.normalize(self.count_matrix, method='css')
        norm2 = self.normalizer.normalize(self.count_matrix, method='css')
        
        np.testing.assert_array_almost_equal(norm1, norm2)


if __name__ == '__main__':
    unittest.main()