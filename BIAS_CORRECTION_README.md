# Bias Correction Implementation for Metagenomic Sample Grouping

## Overview

This implementation provides robust sequencing bias correction methods for metagenomic sample grouping analysis. The goal is to ensure that sample grouping recommendations are based on genuine biological patterns rather than technical artifacts.

## Key Features Implemented

### 1. K-mer Count Normalization (`core/normalization.py`)

Multiple normalization strategies to reduce technical biases:

- **CSS (Cumulative Sum Scaling)**: Inspired by metagenomeSeq, finds percentile-based scaling factors
- **TMM (Trimmed Mean of M-values)**: Inspired by edgeR, uses trimmed mean of log-fold changes
- **RLE (Relative Log Expression)**: Inspired by DESeq2, uses median of ratios to geometric mean
- **TSS (Total Sum Scaling)**: Simple relative abundance normalization
- **CLR (Centered Log-Ratio)**: Compositional data transformation

### 2. Bias-Aware Distance Metrics (`core/similarity.py`)

Robust distance metrics that minimize technical variation impact:

- **Jensen-Shannon Distance**: Proper metric robust to sampling effects
- **Robust Bray-Curtis**: With outlier trimming to reduce technical artifacts
- **Weighted UniFrac**: Adaptation for k-mer data with optional feature weights
- **Aitchison Distance**: Euclidean distance in log-ratio space for compositional data
- **Hellinger Distance**: Robust to sampling variation
- **Chi-squared Distance**: Less sensitive to rare features

### 3. Batch Effect Correction (`core/batch_correction.py`)

Methods to remove technical variation while preserving biological signal:

- **ComBat Correction**: Empirical Bayes approach for batch effect removal
- **Linear Model Correction**: Regression-based approach for continuous and categorical covariates
- **Automatic Batch Detection**: Statistical tests to identify significant batch effects
- **Confounding Assessment**: Detection of technical-biological variable associations

### 4. Bias Impact Assessment (`core/bias_assessment.py`)

Comprehensive evaluation of bias effects and correction effectiveness:

- **PERMANOVA-based Variance Partitioning**: Quantify technical vs biological variance contributions
- **Before/After Comparison**: Assess correction effectiveness
- **Confounding Risk Assessment**: Identify problematic technical-biological associations
- **Robustness Evaluation**: Bootstrap-based stability assessment

## Usage Examples

### Basic Bias-Aware Analysis

```bash
python bias_aware_grouping_analysis.py \
  --samples samples.txt \
  --metadata metadata.csv \
  --output results/ \
  --correct-batch-effects \
  --technical-covariates sequencing_batch extraction_method \
  --biological-covariates environment treatment
```

### Comprehensive Bias Assessment

```bash
python bias_aware_grouping_analysis.py \
  --samples samples.txt \
  --metadata metadata.csv \
  --output results/ \
  --assess-bias-impact \
  --compare-before-after \
  --normalization css \
  --distance jensen_shannon \
  --technical-covariates batch extraction_method sequencing_depth \
  --biological-covariates environment pH temperature
```

### Custom Normalization and Distance Methods

```bash
python bias_aware_grouping_analysis.py \
  --samples samples.txt \
  --metadata metadata.csv \
  --output results/ \
  --normalization tmm \
  --distance robust_bray_curtis \
  --presence-absence \
  --batch-correction-method combat
```

## Expected Output Files

### Bias Correction Files
- `raw_kmer_counts.csv`: Original k-mer count matrix
- `normalized_kmer_counts.csv`: Normalized count matrix
- `batch_corrected_kmer_counts.csv`: Batch-corrected count matrix (if applied)
- `bias_aware_distance_matrix.csv`: Final distance matrix

### Assessment Files
- `bias_assessment.json`: Technical variance assessment results
- `detected_batch_effects.json`: Batch effect detection results
- `confounding_assessment.json`: Confounding risk analysis
- `correction_comparison.json`: Before/after correction comparison
- `normalization_evaluation.json`: Normalization effectiveness metrics
- `distance_robustness.json`: Distance calculation robustness

### Reports
- `bias_assessment_report.json`: Comprehensive bias assessment summary
- `bias_aware_analysis_report.html`: Complete HTML report with visualizations

## Scientific Rationale

### Normalization Methods

1. **CSS**: Addresses variable library sizes and sequencing depth differences
2. **TMM/RLE**: Handle compositional effects and differential feature abundance
3. **CLR**: Appropriate for compositional data analysis

### Distance Metrics

1. **Jensen-Shannon**: Proper metric with theoretical guarantees for probability distributions
2. **Robust Bray-Curtis**: Reduces impact of technical outliers
3. **Aitchison**: Handles compositional nature of relative abundance data

### Batch Correction

1. **ComBat**: Empirically validated approach from genomics literature
2. **Linear Models**: Flexible approach for multiple covariate types
3. **Variance Partitioning**: Quantifies relative contributions of technical vs biological factors

## Quality Control Recommendations

### Before Analysis
1. Examine raw data for obvious batch effects (PCA plots, sample clustering)
2. Check metadata for missing values and potential confounding
3. Assess sequencing depth and quality distributions

### During Analysis
1. Review bias assessment results for significant technical variance
2. Check confounding assessment for problematic associations
3. Evaluate normalization effectiveness metrics

### After Correction
1. Compare before/after correction results
2. Verify biological signal preservation
3. Check robustness of distance calculations
4. Review final grouping recommendations with bias considerations

## Integration with Existing Workflow

The bias correction modules are designed to integrate seamlessly with the existing sample grouping workflow:

1. Original workflow (`sample_grouping_analysis.py`) remains unchanged
2. New bias-aware workflow (`bias_aware_grouping_analysis.py`) provides enhanced analysis
3. All existing visualization and reporting functionality is preserved
4. Results are compatible with downstream assembly optimization steps

## Performance Considerations

- Normalization methods are optimized for large datasets
- Distance calculations use efficient implementations
- Batch correction scales well with sample size
- Bootstrap robustness assessment can be adjusted for speed vs accuracy

## Validation and Testing

Comprehensive unit tests are included for all bias correction modules:
- `tests/test_normalization.py`: Normalization method validation
- `tests/test_similarity.py`: Distance metric testing
- `tests/test_bias_assessment.py`: Assessment functionality testing

Run tests with: `python -m pytest tests/test_*bias*.py -v`

## Literature References

1. Paulson JN, et al. (2013). Differential abundance analysis for microbial marker-gene surveys. Nature Methods 10:1200-1202.
2. Robinson MD, et al. (2010). A scaling normalization method for differential expression analysis of RNA-seq data. Genome Biology 11:R25.
3. Johnson WE, et al. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. Biostatistics 8:118-127.
4. Aitchison J (1986). The Statistical Analysis of Compositional Data. Chapman & Hall.
5. Lin H, et al. (2014). Analysis of compositions of microbiomes with bias correction. Nature Communications 5:3114.