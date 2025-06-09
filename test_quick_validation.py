#!/usr/bin/env python3
"""
Quick validation test that runs fast for debugging API issues.

This creates minimal synthetic data to test the API without long k-mer calculations.
"""

import pandas as pd
import numpy as np
import tempfile
import logging
from pathlib import Path

from testing import AssemblyOptimizationTester

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_minimal_test_data():
    """Create minimal test data for API validation."""
    
    # Create tiny FASTQ files for API testing
    temp_dir = Path(tempfile.mkdtemp())
    
    sample_data = []
    for i in range(3):  # Just 3 samples
        fastq_file = temp_dir / f"sample_{i}.fastq.gz"
        
        # Create tiny FASTQ content (just 10 reads)
        fastq_content = ""
        for read_num in range(10):
            read_id = f"@read_{read_num}_{i}"
            sequence = "ATCGATCGATCGATCG"  # 16bp sequence
            quality = "IIIIIIIIIIIIIIII"   # High quality
            
            fastq_content += f"{read_id}\n{sequence}\n+\n{quality}\n"
        
        # Write compressed FASTQ
        import gzip
        with gzip.open(fastq_file, 'wt') as f:
            f.write(fastq_content)
        
        sample_data.append({
            'sample_id': f'sample_{i}',
            'fastq_file': str(fastq_file),
            'expected_group': f'group_{i%2}',  # 2 groups
            'condition': 'healthy' if i < 2 else 'disease'
        })
    
    # Create metadata
    metadata = pd.DataFrame(sample_data)
    metadata_file = temp_dir / "metadata.csv"
    metadata.to_csv(metadata_file, index=False)
    
    return {
        'temp_dir': temp_dir,
        'metadata_file': metadata_file,
        'sample_data': sample_data
    }

def test_kmer_api():
    """Test the k-mer distance calculation API."""
    logger.info("Testing k-mer distance API...")
    
    from core.kmer_distance import KmerDistanceCalculator
    
    # Create test data
    test_data = create_minimal_test_data()
    
    try:
        # Initialize calculator
        kmer_calc = KmerDistanceCalculator(k=4)  # Small k for speed
        
        # Prepare sample files
        sample_files = {}
        for sample in test_data['sample_data']:
            sample_files[sample['sample_id']] = sample['fastq_file']
        
        logger.info(f"Testing with {len(sample_files)} samples...")
        
        # Test k-mer profile calculation
        logger.info("Calculating k-mer profiles...")
        profiles = kmer_calc.calculate_kmer_profiles(
            sample_files,
            max_reads_per_sample=10  # Very small for speed
        )
        
        logger.info(f"✅ Profiles calculated for {len(profiles)} samples")
        
        # Test distance matrix calculation
        logger.info("Calculating distance matrix...")
        distance_matrix, sample_names = kmer_calc.calculate_distance_matrix(
            profiles,
            metric='braycurtis'
        )
        
        logger.info(f"✅ Distance matrix shape: {distance_matrix.shape}")
        logger.info(f"✅ Sample names: {sample_names}")
        
        return True, test_data['temp_dir']
        
    except Exception as e:
        logger.error(f"❌ K-mer API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, test_data['temp_dir']

def test_grouping_api():
    """Test the grouping optimization API."""
    logger.info("Testing grouping optimization API...")
    
    from core.grouping_optimizer import GroupingOptimizer
    
    try:
        # Create simple test data
        sample_names = ['sample_0', 'sample_1', 'sample_2']
        distance_matrix = np.array([
            [0.0, 0.1, 0.8],
            [0.1, 0.0, 0.7],
            [0.8, 0.7, 0.0]
        ])
        
        metadata = pd.DataFrame({
            'sample_id': sample_names,
            'condition': ['healthy', 'healthy', 'disease']
        }).set_index('sample_id')
        
        # Test grouping optimizer
        optimizer = GroupingOptimizer()
        optimizer.set_distance_matrix(distance_matrix, sample_names)
        optimizer.set_metadata(metadata)
        
        # Test clustering
        clustering_groups = optimizer.generate_clustering_groupings(
            n_clusters=[2],
            method='hierarchical'
        )
        
        logger.info(f"✅ Clustering generated: {len(clustering_groups)} groupings")
        
        # Test evaluation
        for k, grouping in clustering_groups.items():
            evaluation = optimizer.evaluate_grouping(grouping)
            logger.info(f"✅ Grouping evaluation completed: {evaluation['within_group_distance']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Grouping API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bias_api():
    """Test the bias assessment API."""
    logger.info("Testing bias assessment API...")
    
    from core.bias_assessment import BiasAssessment
    
    try:
        # Create test data
        sample_names = ['sample_0', 'sample_1', 'sample_2']
        distance_matrix = np.array([
            [0.0, 0.1, 0.8],
            [0.1, 0.0, 0.7],
            [0.8, 0.7, 0.0]
        ])
        
        metadata = pd.DataFrame({
            'sample_id': sample_names,
            'condition': ['healthy', 'healthy', 'disease'],
            'batch': ['batch1', 'batch1', 'batch2']
        }).set_index('sample_id')
        
        # Test bias assessment
        bias_assessor = BiasAssessment()
        distance_df = pd.DataFrame(distance_matrix, index=sample_names, columns=sample_names)
        
        confounding_results = bias_assessor.detect_confounding_factors(
            distance_df,
            metadata
        )
        
        logger.info(f"✅ Bias assessment completed: {len(confounding_results)} factors tested")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Bias API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run quick API validation tests."""
    logger.info("Starting quick API validation...")
    
    results = {}
    temp_dir = None
    
    # Test k-mer API
    success, temp_dir = test_kmer_api()
    results['kmer_api'] = success
    
    # Test grouping API
    results['grouping_api'] = test_grouping_api()
    
    # Test bias API
    results['bias_api'] = test_bias_api()
    
    # Clean up
    if temp_dir:
        import shutil
        shutil.rmtree(temp_dir)
    
    # Summary
    print("\n" + "="*50)
    print("API VALIDATION RESULTS")
    print("="*50)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All API tests passed! The toolkit should work on the cluster.")
    else:
        print("\n⚠️  Some API tests failed. Need to fix before running on cluster.")
    
    return all_passed

if __name__ == "__main__":
    success = main()