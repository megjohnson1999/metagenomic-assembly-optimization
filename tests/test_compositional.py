"""Unit tests for compositional data handling."""

import unittest
import numpy as np
import pandas as pd

from core.compositional import CompositionalDataHandler


class TestCompositionalDataHandler(unittest.TestCase):
    """Test cases for CompositionalDataHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = CompositionalDataHandler()
        
        # Create test count data
        np.random.seed(42)
        self.n_samples = 8
        self.n_features = 12
        
        # Create compositional data (counts that represent parts of a whole)
        self.count_data = np.random.poisson(10, (self.n_samples, self.n_features))
        
        # Create DataFrame version
        self.count_df = pd.DataFrame(
            self.count_data,
            index=[f"Sample_{i}" for i in range(self.n_samples)],
            columns=[f"Feature_{i}" for i in range(self.n_features)]
        )
        
        # Create data with some zeros for testing
        self.sparse_data = self.count_data.copy()
        self.sparse_data[self.sparse_data < 3] = 0  # Create sparsity
    
    def test_clr_transformation(self):
        """Test Centered Log-Ratio transformation."""
        transformed = self.handler.transform_compositional_data(
            self.count_data, transformation='clr', pseudo_count=0.5
        )
        
        # Check that transformation preserves sample dimension
        self.assertEqual(transformed.shape, self.count_data.shape)
        
        # Check that CLR sums to zero for each sample (within numerical precision)
        row_sums = np.sum(transformed, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.zeros(self.n_samples))
        
        # Check that finite values are returned
        self.assertTrue(np.all(np.isfinite(transformed)))
    
    def test_alr_transformation(self):
        """Test Additive Log-Ratio transformation."""
        transformed = self.handler.transform_compositional_data(
            self.count_data, transformation='alr', pseudo_count=0.5
        )
        
        # Check that ALR reduces dimensionality by 1
        self.assertEqual(transformed.shape, (self.n_samples, self.n_features - 1))
        
        # Check that finite values are returned
        self.assertTrue(np.all(np.isfinite(transformed)))
    
    def test_ilr_transformation(self):
        """Test Isometric Log-Ratio transformation."""
        transformed = self.handler.transform_compositional_data(
            self.count_data, transformation='ilr', pseudo_count=0.5
        )
        
        # Check that ILR reduces dimensionality by 1
        self.assertEqual(transformed.shape, (self.n_samples, self.n_features - 1))
        
        # Check that finite values are returned
        self.assertTrue(np.all(np.isfinite(transformed)))
    
    def test_proportion_transformation(self):
        """Test simple proportion transformation."""
        transformed = self.handler.transform_compositional_data(
            self.count_data, transformation='proportion'
        )
        
        # Check that transformation preserves shape
        self.assertEqual(transformed.shape, self.count_data.shape)
        
        # Check that each row sums to 1 (proportions)
        row_sums = np.sum(transformed, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(self.n_samples))
        
        # Check that all values are non-negative
        self.assertTrue(np.all(transformed >= 0))
    
    def test_dataframe_input(self):
        """Test that DataFrame input is handled correctly."""
        transformed = self.handler.transform_compositional_data(
            self.count_df, transformation='clr'
        )
        
        # Check that result is DataFrame with correct index
        self.assertIsInstance(transformed, pd.DataFrame)
        self.assertEqual(list(transformed.index), list(self.count_df.index))
        
        # For CLR, columns should be the same
        self.assertEqual(list(transformed.columns), list(self.count_df.columns))
    
    def test_alr_dataframe_columns(self):
        """Test that ALR transformation handles DataFrame columns correctly."""
        transformed = self.handler.transform_compositional_data(
            self.count_df, transformation='alr'
        )
        
        # Should have one fewer column (reference removed)
        self.assertEqual(len(transformed.columns), len(self.count_df.columns) - 1)
        
        # Should exclude last column
        expected_columns = list(self.count_df.columns[:-1])
        self.assertEqual(list(transformed.columns), expected_columns)
    
    def test_calculate_aitchison_distance(self):
        """Test Aitchison distance calculation."""
        distance_matrix, sample_names = self.handler.calculate_aitchison_distance(
            self.count_data, pseudo_count=0.5
        )
        
        # Check matrix properties
        self.assertEqual(distance_matrix.shape, (self.n_samples, self.n_samples))
        self.assertEqual(len(sample_names), self.n_samples)
        
        # Check diagonal is zero
        np.testing.assert_array_almost_equal(np.diag(distance_matrix), np.zeros(self.n_samples))
        
        # Check symmetry
        np.testing.assert_array_almost_equal(distance_matrix, distance_matrix.T)
        
        # Check that all distances are non-negative
        self.assertTrue(np.all(distance_matrix >= 0))
    
    def test_detect_compositional_issues(self):
        """Test compositional data issue detection."""
        metrics = self.handler.detect_compositional_issues(self.sparse_data)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'zero_fraction', 'sparse_samples', 'min_features_per_sample',
            'depth_cv', 'rare_features', 'ubiquitous_features',
            'n_samples', 'n_features'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check metric value ranges
        self.assertTrue(0 <= metrics['zero_fraction'] <= 1)
        self.assertTrue(metrics['sparse_samples'] >= 0)
        self.assertTrue(metrics['min_features_per_sample'] >= 0)
        self.assertTrue(metrics['depth_cv'] >= 0)
        self.assertEqual(metrics['n_samples'], self.n_samples)
        self.assertEqual(metrics['n_features'], self.n_features)
    
    def test_recommend_transformation(self):
        """Test transformation recommendation."""
        recommendation = self.handler.recommend_transformation(self.count_data)
        
        # Check that recommendation contains expected keys
        expected_keys = ['recommended_transformation', 'recommended_pseudo_count', 'reasoning']
        for key in expected_keys:
            self.assertIn(key, recommendation)
        
        # Check that transformation is valid
        valid_transforms = ['clr', 'alr', 'ilr', 'proportion']
        self.assertIn(recommendation['recommended_transformation'], valid_transforms)
        
        # Check that pseudo_count is positive
        self.assertGreater(recommendation['recommended_pseudo_count'], 0)
        
        # Check that reasoning is provided
        self.assertIsInstance(recommendation['reasoning'], list)
        self.assertGreater(len(recommendation['reasoning']), 0)
    
    def test_sparse_data_recommendation(self):
        """Test recommendation for sparse data."""
        # Create very sparse data
        very_sparse_data = np.random.poisson(1, (10, 20))
        very_sparse_data[very_sparse_data < 1] = 0  # Make it very sparse
        
        recommendation = self.handler.recommend_transformation(very_sparse_data)
        
        # Should recommend proportion transformation for very sparse data
        if np.mean(very_sparse_data == 0) > 0.8:
            self.assertEqual(recommendation['recommended_transformation'], 'proportion')
    
    def test_evaluate_transformation_quality(self):
        """Test transformation quality evaluation."""
        # Apply CLR transformation
        transformed = self.handler.transform_compositional_data(
            self.count_data, transformation='clr'
        )
        
        # Evaluate quality
        quality_metrics = self.handler.evaluate_transformation_quality(
            self.count_data, transformed
        )
        
        # Check that expected metrics are present
        expected_metrics = ['variance_ratio', 'outlier_fraction', 'effective_dimensionality']
        for metric in expected_metrics:
            self.assertIn(metric, quality_metrics)
        
        # Check metric value ranges
        self.assertGreater(quality_metrics['variance_ratio'], 0)
        self.assertTrue(0 <= quality_metrics['outlier_fraction'] <= 1)
        self.assertGreater(quality_metrics['effective_dimensionality'], 0)
    
    def test_validation_negative_values(self):
        """Test that negative values raise an error."""
        negative_data = self.count_data.copy()
        negative_data[0, 0] = -1
        
        with self.assertRaises(ValueError):
            self.handler.transform_compositional_data(negative_data)
    
    def test_zero_samples_handling(self):
        """Test handling of samples with zero total counts."""
        zero_sample_data = self.count_data.copy()
        zero_sample_data[0, :] = 0  # Make first sample all zeros
        
        # Should handle gracefully with warning
        transformed = self.handler.transform_compositional_data(
            zero_sample_data, transformation='clr'
        )
        
        # Should still return valid shape
        self.assertEqual(transformed.shape, zero_sample_data.shape)
    
    def test_helmert_matrix_creation(self):
        """Test Helmert matrix creation for ILR."""
        n_features = 5
        helmert = self.handler._create_helmert_matrix(n_features)
        
        # Check shape
        self.assertEqual(helmert.shape, (n_features, n_features - 1))
        
        # Check orthogonality (approximately)
        gram_matrix = helmert.T @ helmert
        expected_gram = np.eye(n_features - 1)
        np.testing.assert_array_almost_equal(gram_matrix, expected_gram, decimal=10)
    
    def test_invalid_transformation(self):
        """Test that invalid transformation raises error."""
        with self.assertRaises(ValueError):
            self.handler.transform_compositional_data(
                self.count_data, transformation='invalid_method'
            )
    
    def test_identical_samples(self):
        """Test transformation with identical samples."""
        # Create data with identical samples
        identical_data = np.tile(self.count_data[0, :], (3, 1))
        
        transformed = self.handler.transform_compositional_data(
            identical_data, transformation='clr'
        )
        
        # All transformed samples should be identical
        for i in range(1, 3):
            np.testing.assert_array_almost_equal(transformed[0, :], transformed[i, :])


if __name__ == '__main__':
    unittest.main()