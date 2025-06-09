#!/usr/bin/env python3
"""Command-line interface for sample grouping analysis."""

import argparse
import logging
import sys
from pathlib import Path
import json
import pandas as pd
from typing import Dict, List, Tuple, Union

from core.kmer_distance import KmerDistanceCalculator
from core.metadata_analyzer import MetadataAnalyzer
from core.grouping_optimizer import GroupingOptimizer
from visualization.visualizer import SampleGroupingVisualizer


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_sample_list(sample_list_file: Path) -> Dict[str, Union[Path, Tuple[Path, Path]]]:
    """Parse sample list file to get sample names and file paths.
    
    Expected format (tab-separated):
    sample_name    forward_reads    [reverse_reads]
    
    Args:
        sample_list_file: Path to sample list file
        
    Returns:
        Dictionary mapping sample names to file paths
    """
    samples = {}
    
    with open(sample_list_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t')
            
            if len(parts) < 2:
                logging.warning(f"Skipping line {line_num}: insufficient columns")
                continue
            
            sample_name = parts[0]
            
            if len(parts) == 2:
                # Single-end reads
                samples[sample_name] = Path(parts[1])
            else:
                # Paired-end reads
                samples[sample_name] = (Path(parts[1]), Path(parts[2]))
    
    logging.info(f"Loaded {len(samples)} samples from {sample_list_file}")
    return samples


def main():
    parser = argparse.ArgumentParser(
        description='Analyze sample groupings for metagenomic assembly optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with sample list and metadata
  %(prog)s -s samples.txt -m metadata.csv -o results/
  
  # Specify k-mer size and distance metric
  %(prog)s -s samples.txt -m metadata.csv -o results/ -k 5 --metric cosine
  
  # Limit reads per sample for faster analysis
  %(prog)s -s samples.txt -m metadata.csv -o results/ --max-reads 100000
  
  # Use specific metadata variables for grouping
  %(prog)s -s samples.txt -m metadata.csv -o results/ --group-by environment treatment
        """
    )
    
    # Required arguments
    parser.add_argument('-s', '--samples', required=True, type=Path,
                       help='Sample list file (tab-separated: sample_name, forward_reads, [reverse_reads])')
    parser.add_argument('-m', '--metadata', required=True, type=Path,
                       help='Metadata file (CSV or TSV format)')
    parser.add_argument('-o', '--output', required=True, type=Path,
                       help='Output directory for results')
    
    # K-mer analysis options
    parser.add_argument('-k', '--kmer-size', type=int, default=4,
                       help='K-mer size for distance calculation (default: 4)')
    parser.add_argument('--metric', choices=['braycurtis', 'euclidean', 'cosine', 'jaccard'],
                       default='braycurtis',
                       help='Distance metric to use (default: braycurtis)')
    parser.add_argument('--max-reads', type=int,
                       help='Maximum reads to process per sample (default: all)')
    
    # Metadata analysis options
    parser.add_argument('--sample-id-column', default='sample_id',
                       help='Column name for sample identifiers in metadata (default: sample_id)')
    parser.add_argument('--correlation-method', choices=['mantel', 'anosim'], default='mantel',
                       help='Method for metadata correlation analysis (default: mantel)')
    parser.add_argument('--permutations', type=int, default=999,
                       help='Number of permutations for significance testing (default: 999)')
    
    # Grouping options
    parser.add_argument('--group-by', nargs='+',
                       help='Specific metadata variables to use for grouping')
    parser.add_argument('--min-samples-per-group', type=int, default=3,
                       help='Minimum samples required per group (default: 3)')
    parser.add_argument('--max-clusters', type=int, default=10,
                       help='Maximum number of clusters to try (default: 10)')
    parser.add_argument('--clustering-method', choices=['kmeans', 'hierarchical'], default='kmeans',
                       help='Clustering method to use (default: kmeans)')
    
    # Other options
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level for statistical tests (default: 0.05)')
    parser.add_argument('--n-processes', type=int,
                       help='Number of processes for parallel computation (default: all CPUs)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Load samples
        logger.info("Loading sample information...")
        sample_files = parse_sample_list(args.samples)
        
        # Step 2: Calculate k-mer distances
        logger.info(f"Calculating {args.kmer_size}-mer distances...")
        kmer_calc = KmerDistanceCalculator(k=args.kmer_size, n_processes=args.n_processes)
        kmer_profiles = kmer_calc.calculate_kmer_profiles(
            sample_files, 
            max_reads_per_sample=args.max_reads
        )
        
        distance_matrix, sample_names = kmer_calc.calculate_distance_matrix(
            kmer_profiles, 
            metric=args.metric
        )
        
        # Save distance matrix
        dist_df = pd.DataFrame(distance_matrix, index=sample_names, columns=sample_names)
        dist_df.to_csv(args.output / 'distance_matrix.csv')
        logger.info(f"Saved distance matrix to {args.output / 'distance_matrix.csv'}")
        
        # Step 3: Load and analyze metadata
        logger.info("Analyzing metadata correlations...")
        metadata_analyzer = MetadataAnalyzer()
        metadata = metadata_analyzer.load_metadata(args.metadata, args.sample_id_column)
        metadata_analyzer.set_distance_matrix(distance_matrix, sample_names)
        
        correlation_results = metadata_analyzer.analyze_correlations(
            method=args.correlation_method,
            permutations=args.permutations
        )
        
        # Save correlation results
        with open(args.output / 'metadata_correlations.json', 'w') as f:
            json.dump(correlation_results, f, indent=2)
        
        # Identify significant variables
        significant_vars = metadata_analyzer.identify_significant_variables(
            correlation_results, 
            alpha=args.alpha
        )
        
        logger.info(f"Found {len(significant_vars)} significant metadata variables")
        for var_name, var_results in significant_vars:
            logger.info(f"  - {var_name}: p={var_results['p_value']:.4f}, "
                       f"statistic={var_results['statistic']:.3f}")
        
        # Step 4: Generate and evaluate groupings
        logger.info("Evaluating grouping strategies...")
        grouping_optimizer = GroupingOptimizer()
        grouping_optimizer.set_distance_matrix(distance_matrix, sample_names)
        grouping_optimizer.set_metadata(metadata)
        
        grouping_evaluations = {}
        
        # Evaluate metadata-based groupings
        if args.group_by:
            # Use specified variables
            variables_to_test = args.group_by
        else:
            # Use significant variables
            variables_to_test = [var[0] for var in significant_vars[:5]]  # Top 5
        
        for var_name in variables_to_test:
            if var_name in metadata.columns:
                try:
                    grouping = grouping_optimizer.generate_metadata_groupings(
                        var_name, 
                        min_samples_per_group=args.min_samples_per_group
                    )
                    
                    if grouping:
                        evaluation = grouping_optimizer.evaluate_grouping(grouping)
                        grouping_evaluations[f"metadata_{var_name}"] = evaluation
                        
                        # Save grouping
                        with open(args.output / f'grouping_{var_name}.json', 'w') as f:
                            json.dump(grouping, f, indent=2)
                        
                        logger.info(f"Evaluated grouping by {var_name}: "
                                   f"silhouette={evaluation.get('silhouette_score', 'N/A'):.3f}")
                except Exception as e:
                    logger.warning(f"Failed to evaluate grouping by {var_name}: {e}")
        
        # Evaluate clustering-based groupings
        logger.info("Finding optimal number of clusters...")
        optimal_k, clustering_results = grouping_optimizer.optimize_grouping_number(
            min_clusters=2,
            max_clusters=args.max_clusters,
            method=args.clustering_method
        )
        
        for k, evaluation in clustering_results.items():
            grouping_evaluations[f"clustering_k{k}"] = evaluation
        
        logger.info(f"Optimal number of clusters: {optimal_k}")
        
        # Save evaluation results
        with open(args.output / 'grouping_evaluations.json', 'w') as f:
            json.dump(grouping_evaluations, f, indent=2, default=str)
        
        # Step 5: Generate recommendation
        logger.info("Generating assembly strategy recommendation...")
        recommendation = grouping_optimizer.recommend_assembly_strategy(grouping_evaluations)
        
        with open(args.output / 'recommendation.json', 'w') as f:
            json.dump(recommendation, f, indent=2)
        
        logger.info(f"\nRECOMMENDATION: {recommendation['strategy'].upper()} assembly")
        logger.info(f"Reason: {recommendation['reason']}")
        logger.info(f"Confidence: {recommendation['confidence']:.1%}")
        
        # Step 6: Generate visualizations and report
        logger.info("Generating visualizations and report...")
        visualizer = SampleGroupingVisualizer(figure_dir=args.output / 'figures')
        
        # Generate report
        visualizer.generate_summary_report(
            distance_matrix=distance_matrix,
            sample_names=sample_names,
            correlation_results=correlation_results,
            grouping_evaluations=grouping_evaluations,
            recommendation=recommendation,
            output_path=args.output / 'report.html'
        )
        
        logger.info(f"\nAnalysis complete! Results saved to {args.output}")
        logger.info(f"View the full report at: {args.output / 'report.html'}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()