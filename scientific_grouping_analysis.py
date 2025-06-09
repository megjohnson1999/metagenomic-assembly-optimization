#!/usr/bin/env python3
"""Enhanced scientific sample grouping analysis with validity improvements.

This script provides a streamlined workflow for metagenomic sample grouping analysis
with built-in scientific validity improvements including depth normalization,
compositional data handling, and confounding factor detection.
"""

import argparse
import logging
import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.kmer_distance import KmerDistanceCalculator
from core.metadata_analyzer import MetadataAnalyzer
from core.grouping_optimizer import GroupingOptimizer
from core.visualization import SampleGroupingVisualizer
from core.normalization import KmerNormalizer
from core.compositional import CompositionalDataHandler
from core.bias_assessment import BiasAssessment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_argument_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Enhanced scientific sample grouping analysis for metagenomics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with depth normalization
  python scientific_grouping_analysis.py --samples samples.txt --metadata metadata.csv --output results/ --normalize-depth

  # Full analysis with compositional data handling
  python scientific_grouping_analysis.py --samples samples.txt --metadata metadata.csv --output results/ --handle-compositional --normalize-depth

  # Analysis with confounding factor detection
  python scientific_grouping_analysis.py --samples samples.txt --metadata metadata.csv --output results/ --detect-confounding --technical-vars batch extraction_method
        """
    )
    
    # Required arguments
    parser.add_argument('--samples', required=True,
                       help='File listing sample paths (one per line)')
    parser.add_argument('--metadata', required=True,
                       help='CSV file with sample metadata')
    parser.add_argument('--output', required=True,
                       help='Output directory for results')
    
    # Scientific validity options
    parser.add_argument('--normalize-depth', action='store_true',
                       help='Apply depth normalization to account for sequencing depth variation')
    parser.add_argument('--depth-method', default='scaling',
                       choices=['subsampling', 'scaling', 'rarefaction'],
                       help='Method for depth normalization (default: scaling)')
    parser.add_argument('--handle-compositional', action='store_true',
                       help='Apply compositional data transformations')
    parser.add_argument('--compositional-transform', default='clr',
                       choices=['clr', 'alr', 'ilr', 'proportion'],
                       help='Compositional transformation method (default: clr)')
    parser.add_argument('--detect-confounding', action='store_true',
                       help='Detect potential confounding factors')
    parser.add_argument('--technical-vars', nargs='*',
                       help='List of known technical variables')
    parser.add_argument('--biological-vars', nargs='*',
                       help='List of known biological variables')
    
    # Analysis parameters
    parser.add_argument('--kmer-size', type=int, default=21,
                       help='K-mer size for analysis (default: 21)')
    parser.add_argument('--distance-metric', default='bray_curtis',
                       choices=['bray_curtis', 'euclidean', 'cosine', 'jaccard', 'aitchison'],
                       help='Distance metric for sample comparison')
    parser.add_argument('--normalization', default='css',
                       choices=['css', 'tmm', 'rle', 'tss', 'clr'],
                       help='Normalization method (default: css)')
    parser.add_argument('--min-count', type=int, default=10,
                       help='Minimum k-mer count threshold')
    parser.add_argument('--subsample-reads', type=int,
                       help='Subsample reads for faster analysis')
    
    # Output options
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate comprehensive HTML report')
    parser.add_argument('--save-intermediate', action='store_true',
                       help='Save intermediate results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser


def load_sample_list(samples_file: str) -> List[str]:
    """Load list of sample file paths."""
    with open(samples_file, 'r') as f:
        samples = [line.strip() for line in f if line.strip()]
    
    # Validate sample files exist
    missing_files = [s for s in samples if not os.path.exists(s)]
    if missing_files:
        raise FileNotFoundError(f"Sample files not found: {missing_files}")
    
    logger.info(f"Loaded {len(samples)} sample file paths")
    return samples


def load_metadata(metadata_file: str) -> pd.DataFrame:
    """Load and validate sample metadata."""
    try:
        metadata = pd.read_csv(metadata_file, index_col=0)
    except Exception as e:
        raise ValueError(f"Error loading metadata file: {e}")
    
    logger.info(f"Loaded metadata for {len(metadata)} samples with {len(metadata.columns)} variables")
    return metadata


def calculate_kmer_profiles(samples: List[str], kmer_size: int, min_count: int,
                          subsample_reads: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate k-mer profiles for all samples."""
    logger.info("Calculating k-mer profiles...")
    
    calculator = KmerDistanceCalculator(kmer_size=kmer_size)
    
    kmer_profiles = {}
    sample_names = []
    
    for i, sample_path in enumerate(samples):
        sample_name = os.path.basename(sample_path).split('.')[0]
        sample_names.append(sample_name)
        
        logger.info(f"Processing sample {i+1}/{len(samples)}: {sample_name}")
        
        profile = calculator.calculate_kmer_profiles(
            [sample_path], 
            subsample_reads=subsample_reads
        )[sample_name]
        
        kmer_profiles[sample_name] = profile
    
    # Convert to DataFrame
    all_kmers = set()
    for profile in kmer_profiles.values():
        all_kmers.update(profile.keys())
    
    count_matrix = pd.DataFrame(0, index=sample_names, columns=sorted(all_kmers))
    
    for sample_name, profile in kmer_profiles.items():
        for kmer, count in profile.items():
            if count >= min_count:
                count_matrix.loc[sample_name, kmer] = count
    
    logger.info(f"Generated k-mer matrix: {count_matrix.shape[0]} samples x {count_matrix.shape[1]} k-mers")
    return count_matrix, sample_names


def apply_scientific_corrections(count_matrix: pd.DataFrame, metadata: pd.DataFrame,
                               args) -> Tuple[pd.DataFrame, Dict]:
    """Apply scientific validity corrections to the data."""
    results = {}
    corrected_matrix = count_matrix.copy()
    
    # 1. Depth heterogeneity assessment and normalization
    if args.normalize_depth:
        logger.info("Applying depth normalization...")
        normalizer = KmerNormalizer()
        
        # Assess depth heterogeneity
        depth_metrics = normalizer.detect_depth_heterogeneity(corrected_matrix)
        results['depth_heterogeneity'] = depth_metrics
        
        # Apply depth normalization
        corrected_matrix = normalizer.normalize_for_depth(
            corrected_matrix, method=args.depth_method
        )
        results['depth_normalization_method'] = args.depth_method
        
    # 2. Compositional data handling
    if args.handle_compositional:
        logger.info("Applying compositional data transformations...")
        comp_handler = CompositionalDataHandler()
        
        # Assess compositional data issues
        comp_issues = comp_handler.detect_compositional_issues(corrected_matrix)
        results['compositional_issues'] = comp_issues
        
        # Get transformation recommendation
        transform_rec = comp_handler.recommend_transformation(corrected_matrix)
        results['transformation_recommendation'] = transform_rec
        
        # Apply transformation (use user choice or recommendation)
        if args.compositional_transform == 'auto':
            transform_method = transform_rec['recommended_transformation']
        else:
            transform_method = args.compositional_transform
            
        corrected_matrix = comp_handler.transform_compositional_data(
            corrected_matrix, transformation=transform_method
        )
        results['compositional_transformation'] = transform_method
        
    # 3. Standard normalization
    logger.info(f"Applying {args.normalization} normalization...")
    normalizer = KmerNormalizer()
    corrected_matrix = normalizer.normalize(corrected_matrix, method=args.normalization)
    
    # 4. Confounding factor detection
    if args.detect_confounding:
        logger.info("Detecting confounding factors...")
        bias_assessor = BiasAssessment()
        
        # Calculate similarity matrix for confounding analysis
        if args.distance_metric == 'aitchison' and args.handle_compositional:
            comp_handler = CompositionalDataHandler()
            similarity_matrix, _ = comp_handler.calculate_aitchison_distance(corrected_matrix)
            similarity_matrix = 1 - similarity_matrix  # Convert distance to similarity
        else:
            from scipy.spatial.distance import pdist, squareform
            distances = pdist(corrected_matrix.values, metric='braycurtis')
            similarity_matrix = 1 - squareform(distances)
            
        similarity_df = pd.DataFrame(similarity_matrix, 
                                   index=corrected_matrix.index,
                                   columns=corrected_matrix.index)
        
        confounding_results = bias_assessor.detect_confounding_factors(
            similarity_df, metadata, technical_variables=args.technical_vars
        )
        results['confounding_factors'] = confounding_results
        
    return corrected_matrix, results


def calculate_sample_distances(corrected_matrix: pd.DataFrame, distance_metric: str,
                             handle_compositional: bool) -> Tuple[np.ndarray, List[str]]:
    """Calculate sample distance matrix using appropriate metric."""
    logger.info(f"Calculating {distance_metric} distances...")
    
    sample_names = list(corrected_matrix.index)
    
    if distance_metric == 'aitchison' and handle_compositional:
        # Use Aitchison distance for compositional data
        comp_handler = CompositionalDataHandler()
        distance_matrix, _ = comp_handler.calculate_aitchison_distance(corrected_matrix)
    else:
        # Use k-mer distance calculator
        calculator = KmerDistanceCalculator()
        distance_matrix = calculator.calculate_distance_matrix(
            corrected_matrix.T.to_dict('dict'), metric=distance_metric
        )
    
    return distance_matrix, sample_names


def run_grouping_analysis(distance_matrix: np.ndarray, sample_names: List[str],
                         metadata: pd.DataFrame) -> Dict:
    """Run sample grouping optimization and metadata analysis."""
    logger.info("Running grouping optimization...")
    
    # Metadata analysis
    metadata_analyzer = MetadataAnalyzer()
    metadata_results = metadata_analyzer.analyze_correlations(
        distance_matrix, metadata, sample_names
    )
    
    # Grouping optimization
    grouping_optimizer = GroupingOptimizer()
    grouping_results = grouping_optimizer.optimize_groupings(
        distance_matrix, sample_names, metadata
    )
    
    # Generate recommendations
    recommendations = grouping_optimizer.recommend_assembly_strategy(
        grouping_results, metadata_results
    )
    
    return {
        'metadata_analysis': metadata_results,
        'grouping_optimization': grouping_results,
        'recommendations': recommendations
    }


def save_results(output_dir: str, count_matrix: pd.DataFrame, 
                corrected_matrix: pd.DataFrame, distance_matrix: np.ndarray,
                sample_names: List[str], analysis_results: Dict,
                scientific_results: Dict, args):
    """Save all analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save count matrices
    if args.save_intermediate:
        count_matrix.to_csv(os.path.join(output_dir, 'raw_kmer_counts.csv'))
    corrected_matrix.to_csv(os.path.join(output_dir, 'corrected_kmer_counts.csv'))
    
    # Save distance matrix
    distance_df = pd.DataFrame(distance_matrix, index=sample_names, columns=sample_names)
    distance_df.to_csv(os.path.join(output_dir, 'sample_distance_matrix.csv'))
    
    # Save scientific validity results
    with open(os.path.join(output_dir, 'scientific_validity_assessment.json'), 'w') as f:
        json.dump(scientific_results, f, indent=2, default=str)
    
    # Save analysis results
    with open(os.path.join(output_dir, 'grouping_analysis_results.json'), 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    visualizer = SampleGroupingVisualizer()
    
    visualizer.create_distance_heatmap(
        distance_matrix, sample_names,
        output_path=os.path.join(output_dir, 'distance_heatmap.png')
    )
    
    if 'grouping_optimization' in analysis_results:
        visualizer.create_grouping_visualization(
            analysis_results['grouping_optimization'],
            output_path=os.path.join(output_dir, 'sample_groupings.png')
        )
    
    logger.info(f"Results saved to {output_dir}")


def generate_summary_report(output_dir: str, scientific_results: Dict, 
                          analysis_results: Dict, args) -> str:
    """Generate a summary report of the analysis."""
    report_lines = [
        "# Enhanced Scientific Sample Grouping Analysis Report\n",
        f"**Analysis Parameters:**",
        f"- K-mer size: {args.kmer_size}",
        f"- Distance metric: {args.distance_metric}",
        f"- Normalization: {args.normalization}",
        ""
    ]
    
    # Scientific validity assessment
    report_lines.extend([
        "## Scientific Validity Assessment\n"
    ])
    
    if 'depth_heterogeneity' in scientific_results:
        depth_metrics = scientific_results['depth_heterogeneity']
        report_lines.extend([
            "### Sequencing Depth Analysis",
            f"- Depth CV: {depth_metrics['depth_cv']:.2f}",
            f"- Depth range ratio: {depth_metrics['depth_range_ratio']:.1f}",
            f"- Samples below 1K reads: {depth_metrics['samples_below_1000']}",
            f"- Depth normalization applied: {args.normalize_depth}",
            ""
        ])
    
    if 'compositional_issues' in scientific_results:
        comp_issues = scientific_results['compositional_issues']
        report_lines.extend([
            "### Compositional Data Assessment",
            f"- Zero fraction: {comp_issues['zero_fraction']:.1%}",
            f"- Sparse samples: {comp_issues['sparse_samples']}",
            f"- Rare features: {comp_issues['rare_features']}",
            f"- Compositional handling applied: {args.handle_compositional}",
            ""
        ])
    
    if 'confounding_factors' in scientific_results:
        confounding = scientific_results['confounding_factors']
        high_risk_vars = [var for var, results in confounding.items() 
                         if results.get('confounding_risk') == 'high']
        report_lines.extend([
            "### Confounding Factor Analysis",
            f"- Variables tested: {len(confounding)}",
            f"- High-risk confounding factors: {len(high_risk_vars)}",
            f"- High-risk variables: {', '.join(high_risk_vars) if high_risk_vars else 'None'}",
            ""
        ])
    
    # Assembly recommendations
    if 'recommendations' in analysis_results:
        recommendations = analysis_results['recommendations']
        report_lines.extend([
            "## Assembly Strategy Recommendations\n",
            f"**Recommended strategy:** {recommendations.get('strategy', 'Not determined')}",
            ""
        ])
        
        if 'grouping_variables' in recommendations:
            variables = recommendations['grouping_variables']
            report_lines.extend([
                "**Key grouping variables:**"
            ])
            for var in variables[:3]:  # Top 3
                report_lines.append(f"- {var}")
            report_lines.append("")
    
    # Warnings and recommendations
    warnings = []
    
    if 'depth_heterogeneity' in scientific_results:
        depth_cv = scientific_results['depth_heterogeneity']['depth_cv']
        if depth_cv > 0.5 and not args.normalize_depth:
            warnings.append("High depth variability detected. Consider using --normalize-depth")
    
    if 'compositional_issues' in scientific_results:
        zero_frac = scientific_results['compositional_issues']['zero_fraction']
        if zero_frac > 0.7 and not args.handle_compositional:
            warnings.append("High sparsity detected. Consider using --handle-compositional")
    
    if 'confounding_factors' in scientific_results:
        high_risk_vars = [var for var, results in scientific_results['confounding_factors'].items() 
                         if results.get('confounding_risk') == 'high']
        if high_risk_vars:
            warnings.append(f"High-risk confounding factors detected: {', '.join(high_risk_vars)}")
    
    if warnings:
        report_lines.extend([
            "## Warnings and Recommendations\n"
        ])
        for warning in warnings:
            report_lines.append(f"⚠️ {warning}")
        report_lines.append("")
    
    report_content = "\n".join(report_lines)
    
    # Save report
    report_path = os.path.join(output_dir, 'analysis_summary.md')
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    return report_content


def main():
    """Main analysis workflow."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load data
        logger.info("Loading input data...")
        samples = load_sample_list(args.samples)
        metadata = load_metadata(args.metadata)
        
        # Calculate k-mer profiles
        count_matrix, sample_names = calculate_kmer_profiles(
            samples, args.kmer_size, args.min_count, args.subsample_reads
        )
        
        # Apply scientific validity corrections
        corrected_matrix, scientific_results = apply_scientific_corrections(
            count_matrix, metadata, args
        )
        
        # Calculate sample distances
        distance_matrix, sample_names = calculate_sample_distances(
            corrected_matrix, args.distance_metric, args.handle_compositional
        )
        
        # Run grouping analysis
        analysis_results = run_grouping_analysis(distance_matrix, sample_names, metadata)
        
        # Save results
        save_results(args.output, count_matrix, corrected_matrix, distance_matrix,
                    sample_names, analysis_results, scientific_results, args)
        
        # Generate summary report
        report_content = generate_summary_report(
            args.output, scientific_results, analysis_results, args
        )
        
        logger.info("Analysis completed successfully!")
        
        # Print summary to console
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(report_content)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()