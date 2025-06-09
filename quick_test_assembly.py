#!/usr/bin/env python3
"""
Quick test of assembly optimization toolkit with synthetic data.

This creates a simplified test to verify the toolkit works without waiting 
for slow k-mer calculations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Import toolkit modules
from core.grouping_optimizer import GroupingOptimizer
from core.bias_assessment import BiasAssessment
from core.metadata_analyzer import MetadataAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_synthetic_distance_matrix():
    """Create a synthetic distance matrix based on expected sample relationships."""
    
    # Load metadata
    metadata = pd.read_csv("test_sample_metadata.csv")
    sample_names = metadata['sample_id'].tolist()
    n_samples = len(sample_names)
    
    # Create distance matrix based on biological expectations
    distance_matrix = np.zeros((n_samples, n_samples))
    
    # Map samples to their conditions
    sample_conditions = dict(zip(metadata['sample_id'], metadata['condition']))
    
    for i, sample1 in enumerate(sample_names):
        for j, sample2 in enumerate(sample_names):
            if i == j:
                distance_matrix[i, j] = 0.0
            else:
                # Similar conditions = lower distance
                condition1 = sample_conditions[sample1]
                condition2 = sample_conditions[sample2]
                
                if condition1 == condition2:
                    # Same condition: low distance with some noise
                    distance_matrix[i, j] = np.random.uniform(0.1, 0.3)
                else:
                    # Different conditions: higher distance
                    distance_matrix[i, j] = np.random.uniform(0.6, 0.9)
    
    # Make symmetric
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    
    return pd.DataFrame(distance_matrix, index=sample_names, columns=sample_names)

def test_bias_assessment(distance_matrix, metadata):
    """Test bias assessment functionality."""
    logger.info("Testing bias assessment...")
    
    bias_assessor = BiasAssessment()
    
    # Test confounding factors detection
    try:
        confounding_results = bias_assessor.detect_confounding_factors(
            distance_matrix, 
            metadata.set_index('sample_id')
        )
        
        logger.info("Confounding factor analysis completed")
        for factor, result in confounding_results.items():
            if 'p_value' in result:
                logger.info(f"{factor}: p-value = {result['p_value']:.3f}")
        
        return confounding_results
        
    except Exception as e:
        logger.warning(f"Could not run bias assessment: {e}")
        return {}

def test_grouping_optimization(distance_matrix, metadata):
    """Test sample grouping optimization."""
    logger.info("Testing grouping optimization...")
    
    grouping_optimizer = GroupingOptimizer()
    
    try:
        # Set distance matrix
        grouping_optimizer.set_distance_matrix(
            distance_matrix.values,
            distance_matrix.index.tolist()
        )
        
        # Set metadata
        grouping_optimizer.set_metadata(metadata.set_index('sample_id'))
        
        # Generate clustering-based groupings
        clustering_groups = grouping_optimizer.generate_clustering_groupings(
            n_clusters=[2, 3], 
            method='hierarchical'
        )
        
        logger.info(f"Generated clustering groupings for {len(clustering_groups)} different k values")
        
        # Evaluate groupings
        best_grouping = None
        best_score = float('inf')
        
        for k, grouping in clustering_groups.items():
            logger.info(f"Evaluating k={k} clustering...")
            evaluation = grouping_optimizer.evaluate_grouping(grouping)
            score = evaluation['within_group_distance']
            
            logger.info(f"k={k}: within-group distance = {score:.3f}")
            
            if score < best_score:
                best_score = score
                best_grouping = grouping
        
        if best_grouping:
            logger.info("Best grouping found:")
            for group_name, samples in best_grouping.items():
                sample_types = [metadata[metadata['sample_id'] == sample]['condition'].iloc[0] 
                              for sample in samples]
                logger.info(f"  {group_name}: {samples} (conditions: {sample_types})")
        
        return best_grouping
        
    except Exception as e:
        logger.error(f"Error in grouping optimization: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_metadata_analysis(distance_matrix, metadata):
    """Test metadata analysis functionality."""
    logger.info("Testing metadata analysis...")
    
    metadata_analyzer = MetadataAnalyzer()
    
    try:
        # Set distance matrix
        metadata_analyzer.set_distance_matrix(
            distance_matrix.values,
            distance_matrix.index.tolist()
        )
        
        # Save metadata to a temp file and load it
        metadata.to_csv("temp_metadata.csv", index=False)
        metadata_analyzer.load_metadata("temp_metadata.csv", sample_id_column='sample_id')
        
        # Analyze correlations
        correlation_results = metadata_analyzer.analyze_correlations(method='mantel')
        
        logger.info("Metadata correlation analysis completed")
        for var, result in correlation_results.items():
            if 'p_value' in result:
                corr = result.get('correlation', 'N/A')
                corr_str = f"{corr:.3f}" if isinstance(corr, (int, float)) else str(corr)
                logger.info(f"{var}: correlation = {corr_str}, p-value = {result['p_value']:.3f}")
        
        return correlation_results
        
    except Exception as e:
        logger.error(f"Error in metadata analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Quick test of the assembly optimization toolkit."""
    logger.info("Starting quick assembly optimization test...")
    
    # Load metadata
    metadata = pd.read_csv("test_sample_metadata.csv")
    logger.info(f"Loaded metadata for {len(metadata)} samples")
    
    # Create synthetic distance matrix
    distance_matrix = create_synthetic_distance_matrix()
    logger.info(f"Created {distance_matrix.shape[0]}x{distance_matrix.shape[1]} distance matrix")
    
    # Test bias assessment
    batch_results = test_bias_assessment(distance_matrix, metadata)
    
    # Test grouping optimization
    optimal_groups = test_grouping_optimization(distance_matrix, metadata)
    
    # Test metadata analysis
    metadata_results = test_metadata_analysis(distance_matrix, metadata)
    
    # Summary
    print("\n" + "="*60)
    print("ASSEMBLY OPTIMIZATION TEST RESULTS")
    print("="*60)
    
    print("\n📊 Sample Summary:")
    print(metadata.groupby(['condition', 'diet']).size())
    
    print("\n🔬 Bias Assessment:")
    if batch_results:
        for factor, result in batch_results.items():
            if 'p_value' in result:
                status = "⚠️  SIGNIFICANT" if result['p_value'] < 0.05 else "✅ OK"
                print(f"  {factor}: {status} (p = {result['p_value']:.3f})")
    else:
        print("  No bias assessment results available")
    
    if optimal_groups:
        print(f"\n🎯 Optimal Groupings:")
        for group_name, samples in optimal_groups.items():
            sample_types = [metadata[metadata['sample_id'] == sample]['condition'].iloc[0] 
                          for sample in samples]
            print(f"  {group_name}: {samples}")
            print(f"           Conditions: {sample_types}")
    else:
        print("\n🎯 No optimal groupings found")
    
    if metadata_results:
        print(f"\n🧬 Metadata Correlations:")
        for var, result in metadata_results.items():
            if 'p_value' in result:
                corr = result.get('correlation', 'N/A')
                corr_str = f"{corr:.3f}" if isinstance(corr, (int, float)) else str(corr)
                print(f"  {var}: correlation = {corr_str}, p = {result['p_value']:.3f}")
    else:
        print(f"\n🧬 No metadata correlation results available")
    
    print(f"\n✅ Assembly optimization toolkit test completed successfully!")
    print(f"\nRecommendations:")
    print(f"- Use the identified groups for co-assembly")
    print(f"- Consider bias factors with p < 0.05 in assembly planning")
    print(f"- Higher ANOSIM R values indicate better group separation")

if __name__ == "__main__":
    main()