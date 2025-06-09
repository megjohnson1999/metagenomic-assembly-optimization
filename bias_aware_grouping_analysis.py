#!/usr/bin/env python3
"""Bias-aware command-line interface for sample grouping analysis."""

import argparse
import logging
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union

from core.kmer_distance import KmerDistanceCalculator
from core.metadata_analyzer import MetadataAnalyzer
from core.grouping_optimizer import GroupingOptimizer
from core.normalization import KmerNormalizer
from core.similarity import BiasAwareSimilarity
from core.batch_correction import BatchCorrector
from core.bias_assessment import BiasAssessment
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
    """Parse sample list file to get sample names and file paths."""
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
                samples[sample_name] = Path(parts[1])
            else:
                samples[sample_name] = (Path(parts[1]), Path(parts[2]))
    
    logging.info(f"Loaded {len(samples)} samples from {sample_list_file}")
    return samples


def main():
    parser = argparse.ArgumentParser(
        description='Bias-aware analysis of sample groupings for metagenomic assembly optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with bias correction
  %(prog)s -s samples.txt -m metadata.csv -o results/ --correct-batch-effects
  
  # Specify normalization and distance methods
  %(prog)s -s samples.txt -m metadata.csv -o results/ \\
    --normalization css --distance jensen_shannon --technical-covariates batch extraction_method
  
  # Comprehensive bias assessment
  %(prog)s -s samples.txt -m metadata.csv -o results/ \\
    --assess-bias-impact --biological-covariates environment treatment
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
    parser.add_argument('--max-reads', type=int,
                       help='Maximum reads to process per sample (default: all)')
    
    # Bias correction options
    parser.add_argument('--normalization', 
                       choices=['none', 'css', 'tmm', 'rle', 'tss', 'clr'],
                       default='css',
                       help='Normalization method (default: css)')
    parser.add_argument('--distance', 
                       choices=['jensen_shannon', 'robust_bray_curtis', 'weighted_unifrac', 
                               'aitchison', 'hellinger', 'chi_squared', 'braycurtis'],
                       default='jensen_shannon',
                       help='Distance metric (default: jensen_shannon)')
    parser.add_argument('--presence-absence', action='store_true',
                       help='Convert to presence/absence before distance calculation')
    parser.add_argument('--correct-batch-effects', action='store_true',
                       help='Apply batch effect correction')
    parser.add_argument('--batch-correction-method', 
                       choices=['combat', 'linear'],
                       default='combat',
                       help='Batch correction method (default: combat)')
    parser.add_argument('--technical-covariates', nargs='+',
                       help='Technical covariate column names for bias correction')
    parser.add_argument('--biological-covariates', nargs='+',
                       help='Biological covariate column names to preserve')
    
    # Assessment options
    parser.add_argument('--assess-bias-impact', action='store_true',
                       help='Perform comprehensive bias impact assessment')
    parser.add_argument('--compare-before-after', action='store_true',
                       help='Compare results before and after bias correction')
    
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
        # Step 1: Load samples and metadata
        logger.info("Loading sample information and metadata...")
        sample_files = parse_sample_list(args.samples)
        
        # Load metadata
        metadata_analyzer = MetadataAnalyzer()
        metadata = metadata_analyzer.load_metadata(args.metadata, args.sample_id_column)
        
        # Step 2: Calculate k-mer profiles and initial distances
        logger.info(f"Calculating {args.kmer_size}-mer profiles...")
        kmer_calc = KmerDistanceCalculator(k=args.kmer_size, n_processes=args.n_processes)
        kmer_profiles = kmer_calc.calculate_kmer_profiles(
            sample_files, 
            max_reads_per_sample=args.max_reads
        )
        
        # Convert profiles to count matrix
        sample_names = list(kmer_profiles.keys())
        all_kmers = sorted(set().union(*[set(p.keys()) for p in kmer_profiles.values()]))
        
        count_matrix = np.zeros((len(sample_names), len(all_kmers)))
        for i, sample in enumerate(sample_names):
            for j, kmer in enumerate(all_kmers):
                count_matrix[i, j] = kmer_profiles[sample].get(kmer, 0)
        
        count_df = pd.DataFrame(count_matrix, index=sample_names, columns=all_kmers)
        
        # Save raw k-mer counts
        count_df.to_csv(args.output / 'raw_kmer_counts.csv')
        logger.info(f"Saved raw k-mer counts to {args.output / 'raw_kmer_counts.csv'}")
        
        # Step 3: Bias assessment (if requested)
        bias_assessor = BiasAssessment()
        
        if args.assess_bias_impact and args.technical_covariates:
            logger.info("Assessing bias impact...")
            
            # Calculate initial distance matrix
            similarity_calc = BiasAwareSimilarity()
            initial_distances, _ = similarity_calc.calculate_distances(
                count_df, metric='braycurtis'  # Use standard metric for assessment
            )
            
            # Assess technical variance
            technical_variance = bias_assessor.assess_technical_variance(
                initial_distances, metadata, 
                args.technical_covariates,
                args.biological_covariates,
                n_permutations=args.permutations
            )
            
            # Save bias assessment
            with open(args.output / 'bias_assessment.json', 'w') as f:
                json.dump(technical_variance, f, indent=2, default=str)
                
            # Check confounding
            if args.biological_covariates:
                confounding_assessment = bias_assessor.assess_confounding_risk(
                    metadata, args.technical_covariates, args.biological_covariates
                )
                
                with open(args.output / 'confounding_assessment.json', 'w') as f:
                    json.dump(confounding_assessment, f, indent=2, default=str)
        
        # Step 4: Apply normalization
        processed_data = count_df.copy()
        
        if args.normalization != 'none':
            logger.info(f"Applying {args.normalization} normalization...")
            normalizer = KmerNormalizer()
            processed_data = normalizer.normalize(count_df, method=args.normalization)
            
            # Evaluate normalization
            norm_evaluation = normalizer.evaluate_normalization(
                count_matrix, processed_data.values, metadata
            )
            
            with open(args.output / 'normalization_evaluation.json', 'w') as f:
                json.dump(norm_evaluation, f, indent=2)
                
            # Save normalized data
            processed_data.to_csv(args.output / 'normalized_kmer_counts.csv')
            logger.info(f"Saved normalized k-mer counts")
        
        # Step 5: Apply batch correction (if requested)
        if args.correct_batch_effects and args.technical_covariates:
            logger.info(f"Applying {args.batch_correction_method} batch correction...")
            
            batch_corrector = BatchCorrector()
            
            # Detect batch effects first
            batch_effects = batch_corrector.detect_batch_effects(
                processed_data, metadata, 
                args.technical_covariates,
                args.biological_covariates
            )
            
            with open(args.output / 'detected_batch_effects.json', 'w') as f:
                json.dump(batch_effects, f, indent=2, default=str)
            
            # Apply correction if significant effects detected
            significant_effects = [var for var, stats in batch_effects.items() 
                                 if stats.get('p_value', 1.0) < args.alpha]
            
            if significant_effects:
                if args.batch_correction_method == 'combat':
                    # Use the first significant technical variable as batch variable
                    batch_var = significant_effects[0]
                    corrected_data = batch_corrector.apply_combat_correction(
                        processed_data, metadata, batch_var, args.biological_covariates
                    )
                else:
                    corrected_data = batch_corrector.apply_linear_correction(
                        processed_data, metadata,
                        args.technical_covariates, args.biological_covariates
                    )
                
                processed_data = corrected_data
                
                # Save corrected data
                processed_data.to_csv(args.output / 'batch_corrected_kmer_counts.csv')
                logger.info("Batch correction applied and saved")
                
                # Compare before and after correction
                if args.compare_before_after:
                    logger.info("Comparing before and after bias correction...")
                    comparison_results = bias_assessor.compare_before_after_correction(
                        count_df, processed_data, metadata,
                        args.technical_covariates, args.biological_covariates
                    )
                    
                    with open(args.output / 'correction_comparison.json', 'w') as f:
                        json.dump(comparison_results, f, indent=2, default=str)
            else:
                logger.info("No significant batch effects detected, skipping correction")
        
        # Step 6: Calculate bias-aware distances
        logger.info(f"Calculating bias-aware distances using {args.distance} metric...")
        similarity_calc = BiasAwareSimilarity()
        
        # Calculate feature weights if using weighted metrics
        if args.distance == 'weighted_unifrac':
            feature_weights = similarity_calc.calculate_feature_weights(
                processed_data.values, method='variance_stabilizing'
            )
        else:
            feature_weights = None
        
        distance_matrix, final_sample_names = similarity_calc.calculate_distances(
            processed_data, 
            metric=args.distance,
            presence_absence=args.presence_absence,
            weights=feature_weights
        )
        
        # Evaluate distance robustness
        robustness_metrics = similarity_calc.evaluate_distance_robustness(
            processed_data.values, n_bootstrap=100
        )
        
        with open(args.output / 'distance_robustness.json', 'w') as f:
            json.dump(robustness_metrics, f, indent=2)
        
        # Save distance matrix
        dist_df = pd.DataFrame(distance_matrix, 
                              index=final_sample_names, 
                              columns=final_sample_names)
        dist_df.to_csv(args.output / 'bias_aware_distance_matrix.csv')
        logger.info("Saved bias-aware distance matrix")
        
        # Step 7: Metadata correlation analysis
        logger.info("Analyzing metadata correlations...")
        metadata_analyzer.set_distance_matrix(distance_matrix, final_sample_names)
        
        correlation_results = metadata_analyzer.analyze_correlations(
            method=args.correlation_method,
            permutations=args.permutations
        )
        
        with open(args.output / 'metadata_correlations.json', 'w') as f:
            json.dump(correlation_results, f, indent=2)
        
        significant_vars = metadata_analyzer.identify_significant_variables(
            correlation_results, alpha=args.alpha
        )
        
        logger.info(f"Found {len(significant_vars)} significant metadata variables")
        for var_name, var_results in significant_vars:
            logger.info(f"  - {var_name}: p={var_results['p_value']:.4f}, "
                       f"statistic={var_results['statistic']:.3f}")
        
        # Step 8: Generate and evaluate groupings
        logger.info("Evaluating grouping strategies...")
        grouping_optimizer = GroupingOptimizer()
        grouping_optimizer.set_distance_matrix(distance_matrix, final_sample_names)
        grouping_optimizer.set_metadata(metadata)
        
        grouping_evaluations = {}
        
        # Evaluate metadata-based groupings
        variables_to_test = args.group_by or [var[0] for var in significant_vars[:5]]
        
        for var_name in variables_to_test:
            if var_name in metadata.columns:
                try:
                    grouping = grouping_optimizer.generate_metadata_groupings(
                        var_name, min_samples_per_group=args.min_samples_per_group
                    )
                    
                    if grouping:
                        evaluation = grouping_optimizer.evaluate_grouping(grouping)
                        grouping_evaluations[f"metadata_{var_name}"] = evaluation
                        
                        with open(args.output / f'grouping_{var_name}.json', 'w') as f:
                            json.dump(grouping, f, indent=2)
                        
                        logger.info(f"Evaluated grouping by {var_name}: "
                                   f"silhouette={evaluation.get('silhouette_score', 'N/A'):.3f}")
                except Exception as e:
                    logger.warning(f"Failed to evaluate grouping by {var_name}: {e}")
        
        # Evaluate clustering-based groupings
        logger.info("Finding optimal number of clusters...")
        optimal_k, clustering_results = grouping_optimizer.optimize_grouping_number(
            min_clusters=2, max_clusters=args.max_clusters, method=args.clustering_method
        )
        
        for k, evaluation in clustering_results.items():
            grouping_evaluations[f"clustering_k{k}"] = evaluation
        
        logger.info(f"Optimal number of clusters: {optimal_k}")
        
        with open(args.output / 'grouping_evaluations.json', 'w') as f:
            json.dump(grouping_evaluations, f, indent=2, default=str)
        
        # Step 9: Generate recommendation
        logger.info("Generating assembly strategy recommendation...")
        recommendation = grouping_optimizer.recommend_assembly_strategy(grouping_evaluations)
        
        with open(args.output / 'recommendation.json', 'w') as f:
            json.dump(recommendation, f, indent=2)
        
        logger.info(f"\nRECOMMENDATION: {recommendation['strategy'].upper()} assembly")
        logger.info(f"Reason: {recommendation['reason']}")
        logger.info(f"Confidence: {recommendation['confidence']:.1%}")
        
        # Step 10: Generate comprehensive bias assessment report
        if args.assess_bias_impact:
            logger.info("Generating comprehensive bias assessment report...")
            bias_report = bias_assessor.generate_bias_assessment_report()
            
            with open(args.output / 'bias_assessment_report.json', 'w') as f:
                json.dump(bias_report, f, indent=2, default=str)
            
            # Log key recommendations
            logger.info("\nBias Assessment Recommendations:")
            for rec in bias_report.get('recommendations', []):
                logger.info(f"  - {rec}")
        
        # Step 11: Generate visualizations and report
        logger.info("Generating visualizations and comprehensive report...")
        visualizer = SampleGroupingVisualizer(figure_dir=args.output / 'figures')
        
        # Generate enhanced report with bias correction information
        enhanced_report_data = {
            'distance_matrix': distance_matrix,
            'sample_names': final_sample_names,
            'correlation_results': correlation_results,
            'grouping_evaluations': grouping_evaluations,
            'recommendation': recommendation,
            'bias_correction_applied': args.correct_batch_effects,
            'normalization_method': args.normalization,
            'distance_metric': args.distance,
            'robustness_metrics': robustness_metrics
        }
        
        if args.assess_bias_impact:
            enhanced_report_data['bias_assessment'] = bias_report
        
        # Save comprehensive results
        with open(args.output / 'comprehensive_analysis_results.json', 'w') as f:
            json.dump(enhanced_report_data, f, indent=2, default=str)
        
        # Generate HTML report
        visualizer.generate_summary_report(
            distance_matrix=distance_matrix,
            sample_names=final_sample_names,
            correlation_results=correlation_results,
            grouping_evaluations=grouping_evaluations,
            recommendation=recommendation,
            output_path=args.output / 'bias_aware_analysis_report.html'
        )
        
        logger.info(f"\nBias-aware analysis complete! Results saved to {args.output}")
        logger.info(f"View the full report at: {args.output / 'bias_aware_analysis_report.html'}")
        
        # Summary of bias correction effectiveness
        if args.correct_batch_effects and 'correction_comparison' in locals():
            summary = comparison_results.get('_summary', {})
            effectiveness = summary.get('correction_effectiveness', 0)
            logger.info(f"Bias correction effectiveness: {effectiveness:.1%}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()