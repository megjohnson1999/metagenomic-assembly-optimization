"""Unit tests for bias-aware similarity measures."""

import unittest
import numpy as np
import pandas as pd

from core.similarity import BiasAwareSimilarity


class TestBiasAwareSimilarity(unittest.TestCase):
    """Test cases for BiasAwareSimilarity class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.similarity_calc = BiasAwareSimilarity()
        
        # Create test data (samples x features)
        np.random.seed(42)
        self.n_samples = 5
        self.n_features = 10
        
        # Create count data
        self.count_data = np.random.poisson(5, (self.n_samples, self.n_features))
        
        # Create DataFrame version
        self.count_df = pd.DataFrame(
            self.count_data,
            index=[f"Sample_{i}" for i in range(self.n_samples)],
            columns=[f"Feature_{i}" for i in range(self.n_features)]
        )
    
    def test_jensen_shannon_distance(self):
        """Test Jensen-Shannon distance calculation."""
        dist_matrix, sample_names = self.similarity_calc.calculate_distances(
            self.count_data, metric='jensen_shannon'
        )
        
        # Check matrix properties
        self.assertEqual(dist_matrix.shape, (self.n_samples, self.n_samples))
        self.assertEqual(len(sample_names), self.n_samples)
        
        # Check diagonal is zero
        np.testing.assert_array_almost_equal(np.diag(dist_matrix), np.zeros(self.n_samples))
        
        # Check symmetry
        np.testing.assert_array_almost_equal(dist_matrix, dist_matrix.T)
        
        # Check distance values are in valid range [0, 1]
        self.assertTrue(np.all(dist_matrix >= 0))
        self.assertTrue(np.all(dist_matrix <= 1))
    
    def test_robust_bray_curtis_distance(self):
        """Test robust Bray-Curtis distance calculation."""
        dist_matrix, sample_names = self.similarity_calc.calculate_distances(
            self.count_data, metric='robust_bray_curtis'
        )
        
        # Check matrix properties
        self.assertEqual(dist_matrix.shape, (self.n_samples, self.n_samples))
        
        # Check diagonal is zero
        np.testing.assert_array_almost_equal(np.diag(dist_matrix), np.zeros(self.n_samples))
        
        # Check symmetry
        np.testing.assert_array_almost_equal(dist_matrix, dist_matrix.T)
        
        # Check distance values are in valid range [0, 1]
        self.assertTrue(np.all(dist_matrix >= 0))
        self.assertTrue(np.all(dist_matrix <= 1))
    
    def test_weighted_unifrac_distance(self):
        """Test weighted UniFrac-inspired distance calculation."""
        # Create feature weights
        weights = np.random.uniform(0.5, 2.0, self.n_features)
        
        dist_matrix, sample_names = self.similarity_calc.calculate_distances(
            self.count_data, metric='weighted_unifrac', weights=weights
        )
        
        # Check matrix properties
        self.assertEqual(dist_matrix.shape, (self.n_samples, self.n_samples))
        
        # Check diagonal is zero
        np.testing.assert_array_almost_equal(np.diag(dist_matrix), np.zeros(self.n_samples))
        
        # Check symmetry
        np.testing.assert_array_almost_equal(dist_matrix, dist_matrix.T)
    
    def test_aitchison_distance(self):
        """Test Aitchison distance calculation."""
        dist_matrix, sample_names = self.similarity_calc.calculate_distances(
            self.count_data, metric='aitchison'
        )
        
        # Check matrix properties
        self.assertEqual(dist_matrix.shape, (self.n_samples, self.n_samples))
        
        # Check diagonal is zero
        np.testing.assert_array_almost_equal(np.diag(dist_matrix), np.zeros(self.n_samples))
        
        # Check symmetry
        np.testing.assert_array_almost_equal(dist_matrix, dist_matrix.T)
    
    def test_hellinger_distance(self):
        """Test Hellinger distance calculation."""
        dist_matrix, sample_names = self.similarity_calc.calculate_distances(
            self.count_data, metric='hellinger'
        )
        
        # Check matrix properties
        self.assertEqual(dist_matrix.shape, (self.n_samples, self.n_samples))
        
        # Check diagonal is zero
        np.testing.assert_array_almost_equal(np.diag(dist_matrix), np.zeros(self.n_samples))
        
        # Check symmetry
        np.testing.assert_array_almost_equal(dist_matrix, dist_matrix.T)
        
        # Hellinger distance should be in [0, sqrt(2)]
        self.assertTrue(np.all(dist_matrix >= 0))
        self.assertTrue(np.all(dist_matrix <= np.sqrt(2) + 1e-10))
    
    def test_presence_absence_transformation(self):
        """Test presence/absence transformation."""
        dist_matrix, sample_names = self.similarity_calc.calculate_distances(
            self.count_data, metric='jensen_shannon', presence_absence=True
        )
        
        # Check matrix properties
        self.assertEqual(dist_matrix.shape, (self.n_samples, self.n_samples))
        
        # With presence/absence, distances should be different
        dist_matrix_counts, _ = self.similarity_calc.calculate_distances(
            self.count_data, metric='jensen_shannon', presence_absence=False
        )
        
        # Should be different (unless data is already binary)
        self.assertFalse(np.allclose(dist_matrix, dist_matrix_counts))
    
    def test_dataframe_input(self):
        """Test DataFrame input handling."""
        dist_matrix, sample_names = self.similarity_calc.calculate_distances(
            self.count_df, metric='jensen_shannon'
        )
        
        # Check that sample names are preserved
        expected_names = list(self.count_df.index)
        self.assertEqual(sample_names, expected_names)
    
    def test_calculate_feature_weights(self):
        """Test feature weight calculation methods."""
        methods = ['variance_stabilizing', 'prevalence', 'mean_abundance', 'information_content']
        
        for method in methods:
            weights = self.similarity_calc.calculate_feature_weights(
                self.count_data, method=method
            )
            
            # Check weight properties
            self.assertEqual(len(weights), self.n_features)
            self.assertTrue(np.all(weights >= 0))
            self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
    
    def test_evaluate_distance_robustness(self):
        """Test distance robustness evaluation."""
        # First calculate distances to set the metric
        self.similarity_calc.calculate_distances(self.count_data, metric='jensen_shannon')
        
        robustness = self.similarity_calc.evaluate_distance_robustness(
            self.count_data, n_bootstrap=10, sample_fraction=0.8
        )
        
        # Check that metrics are returned
        self.assertIn('mean_correlation', robustness)
        self.assertIn('std_correlation', robustness)
        self.assertIn('min_correlation', robustness)
        self.assertIn('robustness_score', robustness)
        
        # Check value ranges
        self.assertTrue(0 <= robustness['robustness_score'] <= 1)
        self.assertTrue(-1 <= robustness['mean_correlation'] <= 1)
    
    def test_invalid_metric(self):
        """Test that invalid distance metric raises error."""
        with self.assertRaises(ValueError):
            self.similarity_calc.calculate_distances(
                self.count_data, metric='invalid_metric'
            )
    
    def test_invalid_weights(self):
        """Test that invalid weights raise error."""
        wrong_length_weights = np.ones(self.n_features + 1)
        
        with self.assertRaises(ValueError):
            self.similarity_calc.calculate_distances(
                self.count_data, metric='weighted_unifrac', weights=wrong_length_weights
            )
    
    def test_identical_samples(self):
        """Test distance calculation with identical samples."""
        # Create data with identical samples
        identical_data = np.tile(self.count_data[0, :], (3, 1))
        
        dist_matrix, sample_names = self.similarity_calc.calculate_distances(
            identical_data, metric='jensen_shannon'
        )
        
        # All distances should be zero
        np.testing.assert_array_almost_equal(dist_matrix, np.zeros((3, 3)))


if __name__ == '__main__':
    unittest.main()