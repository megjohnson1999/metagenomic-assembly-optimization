#!/usr/bin/env python3
"""
Unit tests for core assembly optimization functions.

These tests validate individual components in isolation to ensure
correctness of algorithms and edge case handling.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List

# Import core modules to test
from core.kmer_distance import KmerDistanceCalculator
from core.grouping_optimizer import GroupingOptimizer
from core.bias_assessment import BiasAssessment
from core.metadata_analyzer import MetadataAnalyzer
from core.similarity import BiasAwareSimilarity
from core.compositional import CompositionalDataHandler
from core.normalization import KmerNormalizer

class TestKmerDistanceCalculator(unittest.TestCase):
    """Test k-mer distance calculation functionality."""
    
    def setUp(self):
        self.calc = KmerDistanceCalculator(k=4)
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_canonical_kmer(self):
        """Test canonical k-mer generation."""
        # Test simple case
        self.assertEqual(self.calc._get_canonical_kmer("ATCG"), "ATCG")
        self.assertEqual(self.calc._get_canonical_kmer("CGAT"), "ATCG")  # Reverse complement
        
        # Test palindromic k-mer
        self.assertEqual(self.calc._get_canonical_kmer("ATAT"), "ATAT")
    
    def test_kmer_counting(self):
        """Test k-mer counting in reads."""
        read = "ATCGATCG"
        kmers = self.calc._count_kmers_in_read(read)
        
        expected_kmers = ["ATCG", "ATCG", "ATCG", "ATCG", "ATCG"]  # Canonical forms
        self.assertEqual(len(kmers), 5)  # 8 - 4 + 1 = 5 k-mers
    
    def test_distance_metrics(self):
        """Test different distance metrics."""
        # Create simple k-mer count vectors
        counts1 = np.array([10, 5, 0, 2])
        counts2 = np.array([8, 3, 1, 4])
        
        # Test Bray-Curtis
        bc_dist = self.calc._calculate_braycurtis_distance(counts1, counts2)
        self.assertGreaterEqual(bc_dist, 0)
        self.assertLessEqual(bc_dist, 1)
        
        # Test Jaccard
        jaccard_dist = self.calc._calculate_jaccard_distance(counts1, counts2)
        self.assertGreaterEqual(jaccard_dist, 0)
        self.assertLessEqual(jaccard_dist, 1)
        
        # Test cosine
        cosine_dist = self.calc._calculate_cosine_distance(counts1, counts2)
        self.assertGreaterEqual(cosine_dist, 0)
        self.assertLessEqual(cosine_dist, 2)  # Cosine can be up to 2
    
    def test_identical_sequences(self):
        """Test that identical sequences have zero distance."""
        counts = np.array([10, 5, 3, 2])
        
        self.assertEqual(self.calc._calculate_braycurtis_distance(counts, counts), 0.0)
        self.assertEqual(self.calc._calculate_jaccard_distance(counts, counts), 0.0)
        self.assertEqual(self.calc._calculate_cosine_distance(counts, counts), 0.0)
    
    def test_empty_sequences(self):
        """Test handling of empty k-mer counts."""
        empty_counts = np.array([0, 0, 0, 0])
        normal_counts = np.array([1, 2, 3, 4])
        
        # Should handle gracefully without NaN
        bc_dist = self.calc._calculate_braycurtis_distance(empty_counts, normal_counts)
        self.assertFalse(np.isnan(bc_dist))
        
        jaccard_dist = self.calc._calculate_jaccard_distance(empty_counts, normal_counts)
        self.assertFalse(np.isnan(jaccard_dist))

class TestGroupingOptimizer(unittest.TestCase):
    """Test sample grouping optimization."""
    
    def setUp(self):
        self.optimizer = GroupingOptimizer()
        
        # Create sample distance matrix
        self.sample_names = ['sample_A', 'sample_B', 'sample_C', 'sample_D']
        self.distance_matrix = np.array([
            [0.0, 0.1, 0.8, 0.9],
            [0.1, 0.0, 0.7, 0.8],
            [0.8, 0.7, 0.0, 0.1],
            [0.9, 0.8, 0.1, 0.0]
        ])
        
        # Sample metadata
        self.metadata = pd.DataFrame({
            'sample_id': self.sample_names,
            'condition': ['healthy', 'healthy', 'disease', 'disease'],
            'batch': ['batch1', 'batch1', 'batch2', 'batch2']
        }).set_index('sample_id')
        
        self.optimizer.set_distance_matrix(self.distance_matrix, self.sample_names)
        self.optimizer.set_metadata(self.metadata)
    
    def test_clustering_groupings(self):
        """Test clustering-based groupings."""
        groupings = self.optimizer.generate_clustering_groupings(
            n_clusters=[2, 3],
            method='hierarchical'
        )
        
        self.assertIn(2, groupings)
        self.assertIn(3, groupings)
        
        # Check that all samples are assigned
        for k, grouping in groupings.items():
            assigned_samples = []
            for group_samples in grouping.values():
                assigned_samples.extend(group_samples)
            self.assertEqual(set(assigned_samples), set(self.sample_names))
    
    def test_metadata_groupings(self):
        """Test metadata-based groupings."""
        grouping = self.optimizer.generate_metadata_groupings('condition')
        
        self.assertIn('healthy', grouping)
        self.assertIn('disease', grouping)
        self.assertEqual(set(grouping['healthy']), {'sample_A', 'sample_B'})
        self.assertEqual(set(grouping['disease']), {'sample_C', 'sample_D'})
    
    def test_grouping_evaluation(self):
        """Test grouping quality evaluation."""
        # Test good grouping (similar samples together)
        good_grouping = {
            'group1': ['sample_A', 'sample_B'],
            'group2': ['sample_C', 'sample_D']
        }
        
        evaluation = self.optimizer.evaluate_grouping(good_grouping)
        
        self.assertIn('within_group_distance', evaluation)
        self.assertIn('between_group_distance', evaluation)
        self.assertIn('silhouette_score', evaluation)
        
        # Within-group distance should be low for good grouping
        self.assertLess(evaluation['within_group_distance'], 0.5)
    
    def test_single_sample_groups(self):
        """Test handling of single-sample groups."""
        single_grouping = {
            'group1': ['sample_A'],
            'group2': ['sample_B'],
            'group3': ['sample_C'],
            'group4': ['sample_D']
        }
        
        evaluation = self.optimizer.evaluate_grouping(single_grouping)
        
        # Should handle gracefully
        self.assertIsInstance(evaluation, dict)
        self.assertIn('within_group_distance', evaluation)

class TestBiasAssessment(unittest.TestCase):
    """Test bias detection and assessment."""
    
    def setUp(self):
        self.bias_assessor = BiasAssessment()
        
        # Create sample data with known bias structure
        np.random.seed(42)
        n_samples = 20
        n_features = 10
        
        # Create biased data
        self.biased_data = np.random.negative_binomial(5, 0.3, (n_samples, n_features))
        
        # Add batch effect to half the samples
        self.biased_data[10:, :] *= 2  # Double counts for second batch
        
        # Create metadata with batch information
        self.metadata = pd.DataFrame({
            'sample_id': [f'sample_{i:02d}' for i in range(n_samples)],
            'batch': ['batch1'] * 10 + ['batch2'] * 10,
            'condition': ['healthy', 'disease'] * 10,
            'sequencing_depth': np.sum(self.biased_data, axis=1)
        }).set_index('sample_id')
        
        # Create distance matrix
        self.distance_matrix = self.bias_assessor._calculate_bray_curtis_matrix(self.biased_data)
    
    def test_technical_variance_assessment(self):
        """Test technical variance detection."""
        variance_result = self.bias_assessor.assess_technical_variance(
            self.biased_data, 
            self.metadata
        )
        
        self.assertIn('variance_by_factor', variance_result)
        self.assertIn('batch', variance_result['variance_by_factor'])
        
        # Batch should show high variance due to our artificial bias
        batch_variance = variance_result['variance_by_factor']['batch']
        self.assertGreater(batch_variance['variance_ratio'], 1.0)
    
    def test_confounding_detection(self):
        """Test confounding factor detection."""
        confounding_results = self.bias_assessor.detect_confounding_factors(
            pd.DataFrame(self.distance_matrix),
            self.metadata
        )
        
        self.assertIsInstance(confounding_results, dict)
        
        # Should detect batch as potential confounder
        if 'batch' in confounding_results:
            batch_result = confounding_results['batch']
            self.assertIn('p_value', batch_result)
    
    def test_permanova(self):
        """Test PERMANOVA implementation."""
        # Create group labels
        groups = np.array([0] * 10 + [1] * 10)
        
        f_stat, p_value = self.bias_assessor._permanova_test(
            self.distance_matrix, 
            groups, 
            permutations=99
        )
        
        self.assertIsInstance(f_stat, float)
        self.assertIsInstance(p_value, float)
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)
    
    def test_no_bias_data(self):
        """Test with unbiased data."""
        # Create unbiased data
        unbiased_data = np.random.negative_binomial(5, 0.3, (20, 10))
        
        variance_result = self.bias_assessor.assess_technical_variance(
            unbiased_data,
            self.metadata
        )
        
        # Should detect less variance for unbiased data
        self.assertIn('variance_by_factor', variance_result)

class TestMetadataAnalyzer(unittest.TestCase):
    """Test metadata analysis functionality."""
    
    def setUp(self):
        self.analyzer = MetadataAnalyzer()
        
        # Create test data
        self.sample_names = [f'sample_{i:02d}' for i in range(12)]
        
        # Create distance matrix with group structure
        self.distance_matrix = np.random.rand(12, 12)
        
        # Make samples 0-3 similar (group 1)
        for i in range(4):
            for j in range(4):
                if i != j:
                    self.distance_matrix[i, j] = np.random.uniform(0.1, 0.3)
        
        # Make samples 4-7 similar (group 2)  
        for i in range(4, 8):
            for j in range(4, 8):
                if i != j:
                    self.distance_matrix[i, j] = np.random.uniform(0.1, 0.3)
        
        # Make samples 8-11 similar (group 3)
        for i in range(8, 12):
            for j in range(8, 12):
                if i != j:
                    self.distance_matrix[i, j] = np.random.uniform(0.1, 0.3)
        
        # Make symmetric
        self.distance_matrix = (self.distance_matrix + self.distance_matrix.T) / 2
        np.fill_diagonal(self.distance_matrix, 0)
        
        # Create metadata
        self.metadata = pd.DataFrame({
            'sample_id': self.sample_names,
            'condition': ['healthy'] * 4 + ['disease'] * 4 + ['control'] * 4,
            'age': [25, 30, 35, 40] * 3,
            'batch': ['batch1'] * 6 + ['batch2'] * 6
        }).set_index('sample_id')
        
        self.analyzer.set_distance_matrix(self.distance_matrix, self.sample_names)
    
    def test_mantel_test(self):
        """Test Mantel test implementation."""
        # Create two correlated distance matrices
        dist1 = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
        dist2 = np.array([[0, 0.9, 2.1], [0.9, 0, 2.8], [2.1, 2.8, 0]])
        
        r, p_value = self.analyzer._mantel_test(dist1, dist2, permutations=99)
        
        self.assertIsInstance(r, float)
        self.assertIsInstance(p_value, float)
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)
    
    def test_anosim(self):
        """Test ANOSIM implementation."""
        # Create distance matrix with clear group structure
        distances = np.array([
            [0.0, 0.1, 0.8, 0.9],
            [0.1, 0.0, 0.7, 0.8], 
            [0.8, 0.7, 0.0, 0.1],
            [0.9, 0.8, 0.1, 0.0]
        ])
        groups = np.array([0, 0, 1, 1])
        
        r, p_value = self.analyzer._anosim(distances, groups, permutations=99)
        
        self.assertIsInstance(r, float)
        self.assertIsInstance(p_value, float)
        self.assertGreaterEqual(r, -1.0)
        self.assertLessEqual(r, 1.0)
    
    def test_correlation_analysis(self):
        """Test correlation analysis with metadata."""
        # Save temporary metadata file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.metadata.to_csv(f.name)
            temp_file = f.name
        
        try:
            # Load metadata and analyze
            self.analyzer.load_metadata(temp_file)
            
            results = self.analyzer.analyze_correlations(method='mantel', permutations=50)
            
            self.assertIsInstance(results, dict)
            
            # Should have results for categorical variables
            if 'condition' in results:
                self.assertIn('p_value', results['condition'])
            
        finally:
            Path(temp_file).unlink()

class TestCompositionalAnalysis(unittest.TestCase):
    """Test compositional data analysis."""
    
    def setUp(self):
        self.handler = CompositionalDataHandler()
        
        # Create compositional test data
        np.random.seed(42)
        self.count_data = np.random.negative_binomial(10, 0.3, (20, 12))
    
    def test_clr_transformation(self):
        """Test centered log-ratio transformation."""
        transformed = self.handler.transform_compositional_data(
            self.count_data, 
            transformation='clr'
        )
        
        # Check shape preservation
        self.assertEqual(transformed.shape, self.count_data.shape)
        
        # CLR should have zero mean for each sample
        row_means = np.mean(transformed, axis=1)
        np.testing.assert_array_almost_equal(row_means, 0, decimal=10)
    
    def test_alr_transformation(self):
        """Test additive log-ratio transformation."""
        transformed = self.handler.transform_compositional_data(
            self.count_data,
            transformation='alr'
        )
        
        # ALR reduces dimensionality by 1
        self.assertEqual(transformed.shape[0], self.count_data.shape[0])
        self.assertEqual(transformed.shape[1], self.count_data.shape[1] - 1)
    
    def test_ilr_transformation(self):
        """Test isometric log-ratio transformation."""
        transformed = self.handler.transform_compositional_data(
            self.count_data,
            transformation='ilr'
        )
        
        # ILR reduces dimensionality by 1
        self.assertEqual(transformed.shape[0], self.count_data.shape[0])
        self.assertEqual(transformed.shape[1], self.count_data.shape[1] - 1)
    
    def test_proportion_transformation(self):
        """Test proportion transformation."""
        proportions = self.handler.transform_compositional_data(
            self.count_data,
            transformation='proportion'
        )
        
        # Proportions should sum to 1 for each sample
        row_sums = np.sum(proportions, axis=1)
        np.testing.assert_array_almost_equal(row_sums, 1.0, decimal=10)
    
    def test_aitchison_distance(self):
        """Test Aitchison distance calculation."""
        distances, sample_names = self.handler.calculate_aitchison_distance(self.count_data)
        
        # Should be square symmetric matrix
        self.assertEqual(distances.shape[0], distances.shape[1])
        self.assertEqual(distances.shape[0], self.count_data.shape[0])
        
        # Distance matrix should be symmetric
        np.testing.assert_array_almost_equal(distances, distances.T)
        
        # Diagonal should be zero
        np.testing.assert_array_almost_equal(np.diag(distances), 0)
    
    def test_compositional_issues_detection(self):
        """Test detection of compositional data issues."""
        issues = self.handler.detect_compositional_issues(self.count_data)
        
        self.assertIn('zero_proportion', issues)
        self.assertIn('sparsity', issues)
        self.assertIn('has_zeros', issues)
        
        # Should be reasonable values
        self.assertGreaterEqual(issues['zero_proportion'], 0.0)
        self.assertLessEqual(issues['zero_proportion'], 1.0)

class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline_mock_data(self):
        """Test complete pipeline with mock data."""
        # Create mock FASTQ data
        from create_test_fastq_data import create_synthetic_reads
        
        # Generate test files
        sample_data = []
        for i in range(6):
            fastq_file = self.temp_dir / f"sample_{i}.fastq.gz"
            
            # Create different organisms for different samples
            if i < 2:
                organisms = {
                    'Bacteroides': {'GC_content': 0.43, 'signature': 'ATCGATCG', 'abundance': 0.6},
                    'Other': {'GC_content': 0.50, 'signature': 'TTAAGGCC', 'abundance': 0.4}
                }
                group = 'bacteroides_rich'
            elif i < 4:
                organisms = {
                    'Firmicutes': {'GC_content': 0.35, 'signature': 'GCTAGCTA', 'abundance': 0.6},
                    'Other': {'GC_content': 0.50, 'signature': 'TTAAGGCC', 'abundance': 0.4}
                }
                group = 'firmicutes_rich'
            else:
                organisms = {
                    'Bacteroides': {'GC_content': 0.43, 'signature': 'ATCGATCG', 'abundance': 0.4},
                    'Firmicutes': {'GC_content': 0.35, 'signature': 'GCTAGCTA', 'abundance': 0.4},
                    'Other': {'GC_content': 0.50, 'signature': 'TTAAGGCC', 'abundance': 0.2}
                }
                group = 'balanced'
            
            create_synthetic_reads(str(fastq_file), n_reads=1000, organisms=organisms)
            
            sample_data.append({
                'sample_id': f'sample_{i}',
                'fastq_file': str(fastq_file),
                'expected_group': group,
                'batch': 'batch1' if i < 3 else 'batch2'
            })
        
        # Create metadata
        metadata = pd.DataFrame(sample_data)
        metadata_file = self.temp_dir / "metadata.csv"
        metadata.to_csv(metadata_file, index=False)
        
        # Test k-mer distance calculation
        kmer_calc = KmerDistanceCalculator(k=10)
        fastq_files = metadata['fastq_file'].tolist()
        
        distance_matrix, sample_names = kmer_calc.calculate_distance_matrix(
            fastq_files, 
            metric='braycurtis',
            max_reads=500
        )
        
        self.assertEqual(distance_matrix.shape, (6, 6))
        self.assertEqual(len(sample_names), 6)
        
        # Test grouping optimization
        grouping_optimizer = GroupingOptimizer()
        grouping_optimizer.set_distance_matrix(distance_matrix, sample_names)
        grouping_optimizer.set_metadata(metadata.set_index('sample_id'))
        
        clustering_groups = grouping_optimizer.generate_clustering_groupings(
            n_clusters=[2, 3],
            method='hierarchical'
        )
        
        self.assertIn(2, clustering_groups)
        self.assertIn(3, clustering_groups)
        
        # Test bias assessment
        bias_assessor = BiasAssessment()
        distance_df = pd.DataFrame(distance_matrix, index=sample_names, columns=sample_names)
        
        confounding_results = bias_assessor.detect_confounding_factors(
            distance_df,
            metadata.set_index('sample_id')
        )
        
        self.assertIsInstance(confounding_results, dict)
        
        # Pipeline should complete without errors
        logger.info("✅ Full pipeline integration test passed")

def run_unit_tests():
    """Run all unit tests."""
    # Create test suite
    test_classes = [
        TestKmerDistanceCalculator,
        TestGroupingOptimizer, 
        TestBiasAssessment,
        TestMetadataAnalyzer,
        TestCompositionalAnalysis,
        TestIntegration
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    success = run_unit_tests()
    
    if success:
        print("\n✅ All unit tests passed!")
    else:
        print("\n❌ Some unit tests failed!")
        exit(1)