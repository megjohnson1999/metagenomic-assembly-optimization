# Metagenomic Assembly Optimization

A comprehensive toolkit for determining optimal assembly strategies for metagenomic datasets through sample grouping analysis.

## Overview

This repository provides tools and standardized approaches for optimizing metagenomic assembly strategies. The core functionality helps researchers determine whether to use direct assembly or hierarchical assembly approaches based on sequence similarity and metadata correlations.

## Features

### Current Implementation

- **K-mer Based Distance Calculation**: Calculate sequence similarity between samples using k-mer profiles
- **Metadata Correlation Analysis**: Test correlations between metadata variables and sequence similarity
- **Grouping Optimization**: Generate and evaluate sample groupings based on metadata or clustering
- **Visualization and Reporting**: Generate comprehensive visualizations and HTML reports
- **Assembly Strategy Recommendation**: Data-driven recommendations for assembly approach

### Planned Features

- SOP Documentation for systematic assembly optimization
- Dataset characterization tools
- Assembly strategy comparison scripts
- Assembly validation tools
- Utility scripts for file conversion and resource estimation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/metagenomic-assembly-optimization.git
cd metagenomic-assembly-optimization

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```bash
# Run sample grouping analysis
python sample_grouping_analysis.py \
    -s samples.txt \
    -m metadata.csv \
    -o results/
```

### Input Files

1. **Sample List File** (tab-separated):
```
sample_name    forward_reads    reverse_reads
sample1        sample1_R1.fastq.gz    sample1_R2.fastq.gz
sample2        sample2_R1.fastq.gz    sample2_R2.fastq.gz
```

2. **Metadata File** (CSV or TSV):
```
sample_id,environment,pH,temperature
sample1,soil,6.5,20
sample2,water,7.2,18
```

### Command Line Options

```bash
# Specify k-mer size and distance metric
python sample_grouping_analysis.py \
    -s samples.txt \
    -m metadata.csv \
    -o results/ \
    -k 5 \
    --metric cosine

# Limit reads for faster analysis
python sample_grouping_analysis.py \
    -s samples.txt \
    -m metadata.csv \
    -o results/ \
    --max-reads 100000

# Use specific metadata variables for grouping
python sample_grouping_analysis.py \
    -s samples.txt \
    -m metadata.csv \
    -o results/ \
    --group-by environment treatment
```

## Output

The analysis generates several output files:

- `distance_matrix.csv`: Pairwise sample distances based on k-mer profiles
- `metadata_correlations.json`: Statistical analysis of metadata-sequence correlations
- `grouping_evaluations.json`: Evaluation metrics for different grouping strategies
- `recommendation.json`: Assembly strategy recommendation with confidence scores
- `report.html`: Comprehensive HTML report with visualizations
- `figures/`: Directory containing all generated plots

## Workflow

1. **Calculate K-mer Distances**: Compute k-mer profiles from FASTQ files and calculate pairwise distances
2. **Analyze Metadata Correlations**: Test which metadata variables correlate with sequence similarity
3. **Evaluate Groupings**: Generate and evaluate different sample grouping strategies
4. **Generate Recommendation**: Provide data-driven recommendation for assembly approach
5. **Create Report**: Generate comprehensive visualizations and summary report

## Python API

```python
from core.kmer_distance import KmerDistanceCalculator
from core.metadata_analyzer import MetadataAnalyzer
from core.grouping_optimizer import GroupingOptimizer
from visualization.visualizer import SampleGroupingVisualizer

# Calculate k-mer distances
kmer_calc = KmerDistanceCalculator(k=4)
profiles = kmer_calc.calculate_kmer_profiles(sample_files)
distance_matrix, sample_names = kmer_calc.calculate_distance_matrix(profiles)

# Analyze metadata correlations
analyzer = MetadataAnalyzer()
analyzer.load_metadata('metadata.csv')
analyzer.set_distance_matrix(distance_matrix, sample_names)
correlations = analyzer.analyze_correlations()

# Optimize groupings
optimizer = GroupingOptimizer()
optimizer.set_distance_matrix(distance_matrix, sample_names)
recommendation = optimizer.recommend_assembly_strategy(grouping_results)
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=core --cov=visualization
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```
[Citation information to be added]
```

## Contact

For questions or issues, please open an issue on GitHub or contact [your.email@example.com].