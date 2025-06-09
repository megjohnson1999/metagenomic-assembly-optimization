"""Unit tests for k-mer distance calculator."""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import gzip

from core.kmer_distance import KmerDistanceCalculator


class TestKmerDistanceCalculator(unittest.TestCase):
    """Test cases for KmerDistanceCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = KmerDistanceCalculator(k=3)
        
    def test_reverse_complement(self):
        """Test reverse complement generation."""
        test_cases = [
            ("ATCG", "CGAT"),
            ("AAAA", "TTTT"),
            ("GCGC", "GCGC"),
            ("ATCGATCG", "CGATCGAT")
        ]
        
        for seq, expected in test_cases:
            result = self.calculator._reverse_complement(seq)
            self.assertEqual(result, expected)
    
    def test_canonical_kmer(self):
        """Test canonical k-mer selection."""
        # Canonical k-mer is the lexicographically smaller
        test_cases = [
            ("AAA", "AAA"),  # AAA < TTT
            ("TTT", "AAA"),  # Reverse complement of TTT is AAA
            ("ACG", "ACG"),  # ACG < CGT
            ("CGT", "ACG"),  # Reverse complement of CGT is ACG
        ]
        
        for kmer, expected in test_cases:
            result = self.calculator._canonical_kmer(kmer)
            self.assertEqual(result, expected)
    
    def test_count_kmers_in_read(self):
        """Test k-mer counting in sequences."""
        sequence = "ATCGATCG"
        # 3-mers: ATC, TCG, CGA, GAT, ATC, TCG
        # Canonical: ATC->ATC, TCG->CGA, CGA->CGA, GAT->ATC, ATC->ATC, TCG->CGA
        
        counts = self.calculator._count_kmers_in_read(sequence)
        
        self.assertEqual(counts['ATC'], 3)  # ATC appears 3 times
        self.assertEqual(counts['CGA'], 3)  # CGA appears 3 times
        self.assertEqual(len(counts), 2)    # Only 2 unique canonical k-mers
    
    def test_braycurtis_distance(self):
        """Test Bray-Curtis distance calculation."""
        u = np.array([1, 2, 3])
        v = np.array([2, 2, 2])
        
        # Manual calculation: sum(|u-v|) / sum(u+v) = 2 / 12 = 0.1667
        expected = 2 / 12
        result = self.calculator._braycurtis_distance(u, v)
        
        self.assertAlmostEqual(result, expected, places=4)
    
    def test_cosine_distance(self):
        """Test cosine distance calculation."""
        u = np.array([1, 0, 0])
        v = np.array([0, 1, 0])
        
        # Orthogonal vectors have cosine similarity 0, distance 1
        result = self.calculator._cosine_distance(u, v)
        self.assertAlmostEqual(result, 1.0, places=4)
        
        # Same vector has cosine similarity 1, distance 0
        result = self.calculator._cosine_distance(u, u)
        self.assertAlmostEqual(result, 0.0, places=4)
    
    def test_jaccard_distance(self):
        """Test Jaccard distance calculation."""
        u = np.array([1, 1, 0, 0])
        v = np.array([1, 0, 1, 0])
        
        # Intersection: 1, Union: 3, Jaccard distance: 1 - 1/3 = 2/3
        expected = 2/3
        result = self.calculator._jaccard_distance(u, v)
        
        self.assertAlmostEqual(result, expected, places=4)
    
    def test_process_fastq_file(self):
        """Test FASTQ file processing."""
        # Create temporary FASTQ file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fastq', delete=False) as f:
            f.write("@read1\n")
            f.write("ATCGATCGATCG\n")
            f.write("+\n")
            f.write("IIIIIIIIIIII\n")
            f.write("@read2\n")
            f.write("GCGCGCGCGCGC\n")
            f.write("+\n")
            f.write("IIIIIIIIIIII\n")
            temp_path = Path(f.name)
        
        try:
            # Process file
            profile = self.calculator._process_fastq_file(temp_path)
            
            # Check that we got a profile
            self.assertIsInstance(profile, dict)
            self.assertTrue(len(profile) > 0)
            
            # Check that frequencies sum to approximately 1
            total_freq = sum(profile.values())
            self.assertAlmostEqual(total_freq, 1.0, places=2)
            
        finally:
            temp_path.unlink()
    
    def test_calculate_distance_matrix(self):
        """Test distance matrix calculation."""
        # Create mock profiles
        profiles = {
            'sample1': {'AAA': 0.5, 'AAT': 0.3, 'ATT': 0.2},
            'sample2': {'AAA': 0.4, 'AAT': 0.4, 'ATT': 0.2},
            'sample3': {'AAA': 0.1, 'AAT': 0.1, 'ATT': 0.8}
        }
        
        dist_matrix, sample_names = self.calculator.calculate_distance_matrix(
            profiles, metric='braycurtis'
        )
        
        # Check matrix properties
        self.assertEqual(dist_matrix.shape, (3, 3))
        self.assertEqual(len(sample_names), 3)
        
        # Check diagonal is zero
        np.testing.assert_array_almost_equal(np.diag(dist_matrix), [0, 0, 0])
        
        # Check symmetry
        np.testing.assert_array_almost_equal(dist_matrix, dist_matrix.T)
        
        # Check that sample3 is more distant from sample1 and sample2
        self.assertGreater(dist_matrix[0, 2], dist_matrix[0, 1])
        self.assertGreater(dist_matrix[1, 2], dist_matrix[0, 1])


if __name__ == '__main__':
    unittest.main()