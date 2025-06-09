"""Unit tests for bias impact assessment."""

import unittest
import numpy as np
import pandas as pd

from core.bias_assessment import BiasAssessment


class TestBiasAssessment(unittest.TestCase):
    """Test cases for BiasAssessment class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.assessor = BiasAssessment()
        
        # Create test data
        np.random.seed(42)
        self.n_samples = 12
        self.n_features = 8
        
        # Create distance matrix with group structure
        # Samples 0-3: Group A, 4-7: Group B, 8-11: Group C
        self.distance_matrix = np.zeros((self.n_samples, self.n_samples))
        
        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                # Same group: smaller distance
                if (i // 4) == (j // 4):
                    dist = np.random.uniform(0.1, 0.3)
                else:
                    # Different group: larger distance
                    dist = np.random.uniform(0.6, 0.9)
                    
                self.distance_matrix[i, j] = dist
                self.distance_matrix[j, i] = dist
        
        # Create metadata with technical and biological variables
        self.metadata = pd.DataFrame({
            'sample_id': [f"Sample_{i}" for i in range(self.n_samples)],
            'batch': ['batch1'] * 4 + ['batch2'] * 4 + ['batch3'] * 4,  # Technical
            'extraction_method': ['method1'] * 6 + ['method2'] * 6,       # Technical
            'environment': ['soil'] * 4 + ['water'] * 4 + ['air'] * 4,   # Biological
            'temperature': np.random.normal(20, 5, self.n_samples),       # Biological (continuous)
            'sequencing_depth': np.random.uniform(1000, 10000, self.n_samples)  # Technical (continuous)
        })
        self.metadata.set_index('sample_id', inplace=True)
        
        self.sample_names = list(self.metadata.index)
        self.technical_vars = ['batch', 'extraction_method', 'sequencing_depth']
        self.biological_vars = ['environment', 'temperature']
    
    def test_assess_technical_variance(self):
        """Test technical variance assessment."""
        variance_results = self.assessor.assess_technical_variance(
            self.distance_matrix, self.metadata,
            self.technical_vars, self.biological_vars,
            n_permutations=99
        )
        
        # Check that results are returned for each variable
        for var in self.technical_vars + self.biological_vars:
            if var in self.metadata.columns:
                self.assertIn(var, variance_results)
                
                # Check result structure
                result = variance_results[var]
                self.assertIn('variance_explained', result)
                self.assertIn('f_statistic', result)
                self.assertIn('p_value', result)
                self.assertIn('variable_type', result)
                
                # Check value ranges
                self.assertTrue(0 <= result['variance_explained'] <= 1)
                self.assertTrue(0 <= result['p_value'] <= 1)
        
        # Check summary
        if '_summary' in variance_results:
            summary = variance_results['_summary']
            self.assertIn('total_technical_variance', summary)
            self.assertIn('n_technical_vars', summary)
    
    def test_permanova_test(self):
        """Test PERMANOVA implementation."""
        # Test with categorical variable
        result = self.assessor._permanova_test(
            self.distance_matrix, self.metadata, 'batch', n_permutations=99
        )
        
        self.assertIn('variance_explained', result)
        self.assertIn('f_statistic', result)
        self.assertIn('p_value', result)
        
        # Variance explained should be reasonable for this structured data
        self.assertGreater(result['variance_explained'], 0.1)
        
        # Test with continuous variable
        result_cont = self.assessor._permanova_test(
            self.distance_matrix, self.metadata, 'temperature', n_permutations=99
        )
        
        self.assertIn('variance_explained', result_cont)
        self.assertIn('p_value', result_cont)
    
    def test_compare_before_after_correction(self):
        """Test before/after correction comparison."""
        # Create "corrected" data by adding noise
        corrected_matrix = self.distance_matrix + np.random.normal(0, 0.05, self.distance_matrix.shape)
        corrected_matrix = np.maximum(0, corrected_matrix)  # Keep non-negative
        
        # Make symmetric
        corrected_matrix = (corrected_matrix + corrected_matrix.T) / 2
        np.fill_diagonal(corrected_matrix, 0)
        
        comparison = self.assessor.compare_before_after_correction(
            self.distance_matrix, corrected_matrix, self.metadata,
            self.technical_vars, self.biological_vars
        )
        
        # Check that comparison results are returned
        for var in self.technical_vars:
            if var in self.metadata.columns and var in comparison:
                result = comparison[var]
                self.assertIn('original_variance', result)
                self.assertIn('corrected_variance', result)
                self.assertIn('absolute_reduction', result)
                self.assertIn('relative_reduction', result)
        
        # Check biological preservation
        if 'biological_preservation' in comparison:
            bio_results = comparison['biological_preservation']
            for var in self.biological_vars:
                if var in bio_results:
                    self.assertIn('preservation_ratio', bio_results[var])
    
    def test_assess_confounding_risk(self):
        """Test confounding risk assessment."""
        confounding_results = self.assessor.assess_confounding_risk(
            self.metadata, self.technical_vars, self.biological_vars
        )
        
        # Check that results are returned for each technical variable
        for tech_var in self.technical_vars:
            if tech_var in self.metadata.columns and tech_var in confounding_results:
                tech_results = confounding_results[tech_var]
                
                # Check associations with biological variables
                for bio_var in self.biological_vars:
                    if bio_var in tech_results:
                        association = tech_results[bio_var]
                        self.assertIn('effect_size', association)
                        self.assertIn('p_value', association)
                        self.assertIn('test_type', association)
                        
                        # Check value ranges
                        self.assertTrue(0 <= association['effect_size'] <= 1)
                        self.assertTrue(0 <= association['p_value'] <= 1)
        
        # Check high-risk pairs
        if '_high_risk_pairs' in confounding_results:
            high_risk = confounding_results['_high_risk_pairs']
            self.assertIsInstance(high_risk, list)
    
    def test_test_variable_association(self):
        """Test variable association testing."""
        # Test categorical vs categorical
        result_cat_cat = self.assessor._test_variable_association(
            self.metadata['batch'], self.metadata['environment']
        )
        
        self.assertIn('effect_size', result_cat_cat)
        self.assertIn('p_value', result_cat_cat)
        self.assertIn('test_type', result_cat_cat)
        self.assertEqual(result_cat_cat['test_type'], 'chi_squared')
        
        # Test continuous vs continuous
        result_cont_cont = self.assessor._test_variable_association(
            self.metadata['temperature'], self.metadata['sequencing_depth']
        )
        
        self.assertIn('effect_size', result_cont_cont)
        self.assertIn('p_value', result_cont_cont)
        self.assertEqual(result_cont_cont['test_type'], 'pearson_correlation')
        
        # Test categorical vs continuous
        result_mixed = self.assessor._test_variable_association(
            self.metadata['batch'], self.metadata['temperature']
        )
        
        self.assertIn('effect_size', result_mixed)
        self.assertIn('p_value', result_mixed)
        self.assertEqual(result_mixed['test_type'], 'anova')
    
    def test_generate_bias_assessment_report(self):
        """Test comprehensive report generation."""
        # First run some assessments
        self.assessor.assess_technical_variance(
            self.distance_matrix, self.metadata,
            self.technical_vars, self.biological_vars,
            n_permutations=99
        )
        
        self.assessor.assess_confounding_risk(
            self.metadata, self.technical_vars, self.biological_vars
        )
        
        # Generate report
        report = self.assessor.generate_bias_assessment_report()
        
        # Check report structure
        self.assertIn('summary', report)
        self.assertIn('technical_variance_assessment', report)
        self.assertIn('confounding_assessment', report)
        self.assertIn('recommendations', report)
        
        # Check that recommendations are strings
        recommendations = report['recommendations']
        self.assertIsInstance(recommendations, list)
        for rec in recommendations:
            self.assertIsInstance(rec, str)
    
    def test_calculate_bray_curtis_matrix(self):
        """Test Bray-Curtis distance matrix calculation."""
        # Create test count data
        count_data = np.random.poisson(10, (5, 8))
        
        dist_matrix = self.assessor._calculate_bray_curtis_matrix(count_data)
        
        # Check matrix properties
        self.assertEqual(dist_matrix.shape, (5, 5))
        
        # Check diagonal is zero
        np.testing.assert_array_almost_equal(np.diag(dist_matrix), np.zeros(5))
        
        # Check symmetry
        np.testing.assert_array_almost_equal(dist_matrix, dist_matrix.T)
        
        # Check distance values are in valid range [0, 1]
        self.assertTrue(np.all(dist_matrix >= 0))
        self.assertTrue(np.all(dist_matrix <= 1))
    
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        # Create metadata with missing values
        metadata_with_na = self.metadata.copy()
        metadata_with_na.loc[metadata_with_na.index[:3], 'batch'] = np.nan
        
        variance_results = self.assessor.assess_technical_variance(
            self.distance_matrix, metadata_with_na,
            ['batch'], ['environment'],
            n_permutations=99
        )
        
        # Should still work with remaining samples
        if 'batch' in variance_results:
            result = variance_results['batch']
            self.assertIn('variance_explained', result)
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data scenarios."""
        # Create very small dataset
        small_metadata = self.metadata.iloc[:2].copy()
        small_distance_matrix = self.distance_matrix[:2, :2]
        
        variance_results = self.assessor.assess_technical_variance(
            small_distance_matrix, small_metadata,
            ['batch'], ['environment'],
            n_permutations=10
        )
        
        # Should handle gracefully (may return empty results)
        self.assertIsInstance(variance_results, dict)


if __name__ == '__main__':
    unittest.main()