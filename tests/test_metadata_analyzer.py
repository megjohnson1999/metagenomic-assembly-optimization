"""Unit tests for metadata analyzer."""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from core.metadata_analyzer import MetadataAnalyzer


class TestMetadataAnalyzer(unittest.TestCase):
    """Test cases for MetadataAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = MetadataAnalyzer()
        
        # Create test distance matrix
        self.distance_matrix = np.array([
            [0.0, 0.1, 0.5, 0.6],
            [0.1, 0.0, 0.4, 0.5],
            [0.5, 0.4, 0.0, 0.2],
            [0.6, 0.5, 0.2, 0.0]
        ])
        self.sample_names = ['S1', 'S2', 'S3', 'S4']
        
        # Create test metadata
        self.metadata = pd.DataFrame({
            'environment': ['soil', 'soil', 'water', 'water'],
            'pH': [6.5, 6.8, 7.2, 7.5],
            'temperature': [20, 22, 18, 19]
        }, index=self.sample_names)
    
    def test_load_metadata_csv(self):
        """Test loading metadata from CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("sample_id,environment,pH\n")
            f.write("S1,soil,6.5\n")
            f.write("S2,water,7.2\n")
            temp_path = Path(f.name)
        
        try:
            # Load metadata
            df = self.analyzer.load_metadata(temp_path)
            
            # Check loaded data
            self.assertEqual(len(df), 2)
            self.assertIn('environment', df.columns)
            self.assertIn('pH', df.columns)
            self.assertEqual(df.loc['S1', 'environment'], 'soil')
            
        finally:
            temp_path.unlink()
    
    def test_load_metadata_tsv(self):
        """Test loading metadata from TSV file."""
        # Create temporary TSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            f.write("sample_id\tenvironment\tpH\n")
            f.write("S1\tsoil\t6.5\n")
            f.write("S2\twater\t7.2\n")
            temp_path = Path(f.name)
        
        try:
            # Load metadata
            df = self.analyzer.load_metadata(temp_path)
            
            # Check loaded data
            self.assertEqual(len(df), 2)
            self.assertIn('environment', df.columns)
            
        finally:
            temp_path.unlink()
    
    def test_set_distance_matrix(self):
        """Test setting distance matrix."""
        self.analyzer.set_distance_matrix(self.distance_matrix, self.sample_names)
        
        np.testing.assert_array_equal(self.analyzer.distance_matrix, self.distance_matrix)
        self.assertEqual(self.analyzer.sample_names, self.sample_names)
        
        # Test invalid inputs
        with self.assertRaises(ValueError):
            # Non-square matrix
            self.analyzer.set_distance_matrix(np.array([[0, 1], [1, 0], [2, 3]]), ['A', 'B'])
        
        with self.assertRaises(ValueError):
            # Mismatched dimensions
            self.analyzer.set_distance_matrix(self.distance_matrix, ['A', 'B'])
    
    def test_validate_samples(self):
        """Test sample validation between metadata and distance matrix."""
        self.analyzer.metadata = self.metadata
        self.analyzer.set_distance_matrix(self.distance_matrix, self.sample_names)
        
        common_samples, aligned_dist, aligned_meta = self.analyzer._validate_samples()
        
        # All samples should be common
        self.assertEqual(len(common_samples), 4)
        self.assertEqual(set(common_samples), set(self.sample_names))
    
    def test_mantel_test(self):
        """Test Mantel test implementation."""
        # Create two correlated distance matrices
        dist1 = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
        dist2 = np.array([[0, 0.9, 2.1], [0.9, 0, 2.8], [2.1, 2.8, 0]])
        
        r, p_value = self.analyzer._mantel_test(dist1, dist2, permutations=99)
        
        # Should have high correlation
        self.assertGreater(r, 0.9)
        self.assertLess(p_value, 0.05)
    
    def test_anosim(self):
        """Test ANOSIM implementation."""
        # Distance matrix with clear group structure
        distances = np.array([
            [0.0, 0.1, 0.8, 0.9],
            [0.1, 0.0, 0.7, 0.8],
            [0.8, 0.7, 0.0, 0.1],
            [0.9, 0.8, 0.1, 0.0]
        ])
        groups = np.array([0, 0, 1, 1])  # Two groups
        
        r, p_value = self.analyzer._anosim(distances, groups, permutations=99)
        
        # Should have significant group separation
        self.assertGreater(r, 0.5)
        self.assertLess(p_value, 0.05)
    
    def test_analyze_categorical_variable(self):
        """Test categorical variable analysis."""
        self.analyzer.metadata = self.metadata
        self.analyzer.set_distance_matrix(self.distance_matrix, self.sample_names)
        
        # Analyze environment variable
        result = self.analyzer._analyze_categorical_variable(
            self.metadata['environment'], 
            self.distance_matrix,
            'anosim',
            99
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result['type'], 'categorical')
        self.assertEqual(result['n_categories'], 2)
        self.assertIn('p_value', result)
        self.assertIn('statistic', result)
    
    def test_analyze_continuous_variable(self):
        """Test continuous variable analysis."""
        self.analyzer.metadata = self.metadata
        self.analyzer.set_distance_matrix(self.distance_matrix, self.sample_names)
        
        # Analyze pH variable
        result = self.analyzer._analyze_continuous_variable(
            self.metadata['pH'], 
            self.distance_matrix,
            'mantel',
            99
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result['type'], 'continuous')
        self.assertEqual(result['method'], 'mantel')
        self.assertIn('p_value', result)
        self.assertIn('statistic', result)
    
    def test_identify_significant_variables(self):
        """Test identification of significant variables."""
        # Create mock results
        results = {
            'var1': {'p_value': 0.01, 'statistic': 0.8},
            'var2': {'p_value': 0.10, 'statistic': 0.3},
            'var3': {'p_value': 0.03, 'statistic': 0.6},
            'var4': {'p_value': 0.50, 'statistic': 0.1}
        }
        
        significant = self.analyzer.identify_significant_variables(results, alpha=0.05)
        
        # Should have 2 significant variables
        self.assertEqual(len(significant), 2)
        
        # Should be sorted by p-value
        self.assertEqual(significant[0][0], 'var1')
        self.assertEqual(significant[1][0], 'var3')


if __name__ == '__main__':
    unittest.main()