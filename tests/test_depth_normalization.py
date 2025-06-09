"""Unit tests for depth normalization functionality."""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch

from core.normalization import KmerNormalizer


class TestDepthNormalization(unittest.TestCase):
    """Test cases for depth normalization methods in KmerNormalizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.normalizer = KmerNormalizer()
        
        # Create test data with varying depths
        np.random.seed(42)
        self.n_samples = 6
        self.n_features = 10
        
        # Create count data with different sequencing depths
        depths = [1000, 2000, 5000, 8000, 15000, 20000]  # Varying depths
        self.count_data = np.zeros((self.n_samples, self.n_features))
        
        for i, depth in enumerate(depths):
            # Generate counts proportional to depth
            sample_counts = np.random.poisson(depth / self.n_features, self.n_features)
            self.count_data[i, :] = sample_counts
            
        # Create DataFrame version
        self.count_df = pd.DataFrame(
            self.count_data,
            index=[f"Sample_{i}" for i in range(self.n_samples)],
            columns=[f"Feature_{i}" for i in range(self.n_features)]
        )
        
        self.sample_depths = np.sum(self.count_data, axis=1)
    
    def test_detect_depth_heterogeneity(self):
        """Test depth heterogeneity detection."""
        metrics = self.normalizer.detect_depth_heterogeneity(self.count_data)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'min_depth', 'max_depth', 'median_depth', 'mean_depth',
            'depth_cv', 'depth_range_ratio', 'samples_below_1000', 'samples_above_100k'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check metric value ranges and logic
        self.assertGreater(metrics['max_depth'], metrics['min_depth'])
        self.assertGreater(metrics['depth_range_ratio'], 1.0)
        self.assertGreater(metrics['depth_cv'], 0)  # Should have variation
        self.assertEqual(metrics['samples_below_1000'], 0)  # No samples below 1000 in our data
        self.assertEqual(metrics['samples_above_100k'], 0)  # No samples above 100k in our data
    
    def test_subsampling_normalization(self):
        """Test subsampling normalization method."""
        normalized = self.normalizer.normalize_for_depth(
            self.count_data, sample_depths=self.sample_depths, method="subsampling"
        )
        
        # Check that result has same shape
        self.assertEqual(normalized.shape, self.count_data.shape)
        
        # Check that all samples have approximately the same total depth (minimum)
        normalized_depths = np.sum(normalized, axis=1)
        min_original_depth = np.min(self.sample_depths)
        
        # All normalized depths should be <= min_original_depth
        self.assertTrue(np.all(normalized_depths <= min_original_depth + 1))  # +1 for floating point
        
        # Check that counts are reduced or equal
        self.assertTrue(np.all(normalized <= self.count_data))
    
    def test_scaling_normalization(self):
        """Test scaling normalization method."""
        normalized = self.normalizer.normalize_for_depth(
            self.count_data, sample_depths=self.sample_depths, method="scaling"
        )
        
        # Check that result has same shape
        self.assertEqual(normalized.shape, self.count_data.shape)
        
        # Check that all samples have approximately the same total depth (median)
        normalized_depths = np.sum(normalized, axis=1)
        median_original_depth = np.median(self.sample_depths)
        
        # All normalized depths should be close to median
        np.testing.assert_allclose(normalized_depths, median_original_depth, rtol=0.01)
    
    def test_rarefaction_normalization(self):
        """Test rarefaction normalization method."""
        # Mock random functions for reproducible results
        with patch('numpy.random.multinomial') as mock_multinomial:
            # Set up mock to return predictable results
            mock_multinomial.return_value = np.ones(self.n_features, dtype=int)
            
            normalized = self.normalizer.normalize_for_depth(
                self.count_data, sample_depths=self.sample_depths, method="rarefaction"
            )
            
            # Check that result has same shape
            self.assertEqual(normalized.shape, self.count_data.shape)
            
            # Check that finite values are returned
            self.assertTrue(np.all(np.isfinite(normalized)))
    
    def test_dataframe_input_depth_normalization(self):
        """Test that DataFrame input is handled correctly."""
        normalized = self.normalizer.normalize_for_depth(
            self.count_df, method="scaling"
        )
        
        # Check that result is DataFrame with correct structure
        self.assertIsInstance(normalized, pd.DataFrame)
        self.assertEqual(list(normalized.index), list(self.count_df.index))
        self.assertEqual(list(normalized.columns), list(self.count_df.columns))
    
    def test_auto_depth_calculation(self):
        """Test automatic depth calculation when depths not provided."""
        normalized = self.normalizer.normalize_for_depth(
            self.count_data, method="scaling"
        )
        
        # Should work without explicit depths
        self.assertEqual(normalized.shape, self.count_data.shape)
        self.assertTrue(np.all(np.isfinite(normalized)))
    
    def test_zero_depth_handling(self):
        """Test handling of samples with zero depth."""
        zero_depth_data = self.count_data.copy()
        zero_depth_data[0, :] = 0  # Make first sample have zero depth
        
        normalized = self.normalizer.normalize_for_depth(
            zero_depth_data, method="scaling"
        )
        
        # Should handle gracefully
        self.assertEqual(normalized.shape, zero_depth_data.shape)
        
        # Zero depth sample should remain zero
        np.testing.assert_array_equal(normalized[0, :], np.zeros(self.n_features))
    
    def test_depth_similarity_correlation_warning(self):
        """Test that depth-similarity correlation is detected."""
        # Create data where similarity correlates with depth
        correlated_data = np.zeros((4, 5))
        
        # Make samples with similar depths have similar profiles
        correlated_data[0, :] = [10, 20, 5, 15, 10]    # Low depth
        correlated_data[1, :] = [12, 18, 7, 13, 8]     # Low depth, similar profile
        correlated_data[2, :] = [100, 200, 50, 150, 100]  # High depth
        correlated_data[3, :] = [120, 180, 70, 130, 80]   # High depth, similar profile
        
        with patch('core.normalization.logger') as mock_logger:
            self.normalizer.normalize_for_depth(correlated_data, method="scaling")
            
            # Should call logger with correlation info
            self.assertTrue(mock_logger.info.called or mock_logger.warning.called)
    
    def test_high_cv_warning(self):
        """Test that high coefficient of variation triggers warning."""
        # Create data with very high depth variation
        high_var_data = np.array([
            [1, 1, 1, 1, 1],      # Very low depth
            [1000, 1000, 1000, 1000, 1000]  # Very high depth
        ])
        
        with patch('core.normalization.logger') as mock_logger:
            self.normalizer.normalize_for_depth(high_var_data, method="scaling")
            
            # Should issue warning about high heterogeneity
            mock_logger.warning.assert_called()
            warning_message = mock_logger.warning.call_args[0][0]
            self.assertIn("heterogeneity", warning_message.lower())
    
    def test_invalid_depth_method(self):
        """Test that invalid depth normalization method raises error."""
        with self.assertRaises(ValueError):
            self.normalizer.normalize_for_depth(
                self.count_data, method="invalid_method"
            )
    
    def test_pandas_series_depths(self):
        """Test that pandas Series depths are handled correctly."""
        depths_series = pd.Series(self.sample_depths, 
                                index=[f"Sample_{i}" for i in range(self.n_samples)])
        
        normalized = self.normalizer.normalize_for_depth(
            self.count_df, sample_depths=depths_series, method="scaling"
        )
        
        # Should work with Series input
        self.assertIsInstance(normalized, pd.DataFrame)
        self.assertEqual(normalized.shape, self.count_df.shape)
    
    def test_depth_metrics_comprehensive(self):
        """Test comprehensive depth heterogeneity metrics."""
        # Create data with known characteristics
        test_data = np.array([
            [500, 300, 200],      # Below 1000 reads
            [2000, 1500, 1000],   # Normal depth
            [5000, 3000, 2000],   # Normal depth
            [150000, 100000, 50000]  # Above 100k reads
        ])
        
        metrics = self.normalizer.detect_depth_heterogeneity(test_data)
        
        # Check specific counts
        self.assertEqual(metrics['samples_below_1000'], 1)
        self.assertEqual(metrics['samples_above_100k'], 1)
        
        # Check that min/max are correct
        expected_depths = np.sum(test_data, axis=1)
        self.assertEqual(metrics['min_depth'], float(np.min(expected_depths)))
        self.assertEqual(metrics['max_depth'], float(np.max(expected_depths)))
    
    def test_normalization_preserves_proportions(self):
        """Test that scaling normalization preserves relative proportions."""
        # Use simple data to verify proportions
        simple_data = np.array([
            [10, 20, 30],  # Sample 1: total = 60
            [20, 40, 60]   # Sample 2: total = 120 (2x sample 1)
        ])
        
        normalized = self.normalizer.normalize_for_depth(simple_data, method="scaling")
        
        # Proportions within each sample should be preserved
        original_props_1 = simple_data[0, :] / np.sum(simple_data[0, :])
        normalized_props_1 = normalized[0, :] / np.sum(normalized[0, :])
        
        np.testing.assert_allclose(original_props_1, normalized_props_1, rtol=1e-10)
    
    def test_empty_data_handling(self):
        """Test handling of edge cases with empty or minimal data."""
        # Test with single sample
        single_sample = self.count_data[:1, :]
        
        normalized = self.normalizer.normalize_for_depth(single_sample, method="scaling")
        
        # Should handle single sample gracefully
        self.assertEqual(normalized.shape, single_sample.shape)
        
        # Test with single feature
        single_feature = self.count_data[:, :1]
        
        normalized = self.normalizer.normalize_for_depth(single_feature, method="scaling")
        
        # Should handle single feature gracefully
        self.assertEqual(normalized.shape, single_feature.shape)


if __name__ == '__main__':
    unittest.main()