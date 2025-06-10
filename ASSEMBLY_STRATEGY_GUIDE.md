# Assembly Strategy Decision Tree Guide

A systematic framework for selecting optimal assembly strategies for metagenomic datasets.

## Overview

This guide provides a structured decision-making process to help researchers choose between different assembly approaches based on their data characteristics, research goals, and available resources. The framework considers four main assembly strategies:

1. **Individual Assembly**: Each sample assembled separately
2. **Metadata-Guided Co-Assembly**: Samples grouped by biological metadata variables
3. **Similarity-Based Co-Assembly**: Samples grouped by sequence similarity
4. **Pooled Co-Assembly**: All samples assembled together

## Decision Tree Framework

### STEP 1: Data Inventory & Quality Check

**Quick file assessment (5-10 minutes):**

```bash
# Count total samples
find . -name "*.fastq*" | wc -l

# Check file sizes (depth estimation)
ls -lh *.fastq* | head -10

# Verify read types
zcat sample.fastq.gz | head -8  # Check for paired-end format
```

**Decision Points:**

1. **Sample Count Assessment**
   - **< 3 samples** → **Individual Assembly Only**
   - **3-10 samples** → Limited grouping options available
   - **> 10 samples** → Full range of strategies available

2. **Sequencing Depth per Sample**
   - **> 20M reads/sample** → Individual assembly viable
   - **5-20M reads/sample** → Evaluate co-assembly benefits
   - **< 5M reads/sample** → Co-assembly likely necessary

3. **Read Type Considerations**
   - **Single-end reads** → Co-assembly preferred (assembly challenges)
   - **Paired-end reads** → All strategies viable
   - **Long reads (ONT/PacBio)** → Different workflow entirely
   - **Read length < 100bp** → Co-assembly recommended

### STEP 2: Community Structure Assessment

**Choose your assessment approach based on available resources:**

#### Option A: Biological Knowledge-Based (2 minutes)
Answer based on your experimental design:

1. **Are samples from similar environments?**
   - YES (same body site, similar soil types) → Consider co-assembly
   - NO (gut vs soil, different hosts) → Individual assembly

2. **Do you expect shared species between samples?**
   - YES → Co-assembly beneficial
   - NO → Individual assembly
   - UNSURE → Default to co-assembly (safer for discovery)

#### Option B: Rapid Computational Assessment (30-60 minutes)
Choose one of these distance matrix approaches:

**Fast Sketching Approach (Recommended for >50 samples):**
```bash
# Sourmash sketching
for sample in *.fastq*; do
    sourmash sketch dna -p k=21,k=31 --scaled=1000 $sample -o ${sample}.sig
done
sourmash compare *.sig -o distance_matrix.csv
```

**K-mer Profile Approach (Most assembly-relevant):**
```python
from core.kmer_distance import KmerDistanceCalculator
calc = KmerDistanceCalculator(k=21, canonical=True)
profiles = {sample: calc.calculate_kmer_profile(sample) for sample in samples}
distance_matrix = calc.calculate_pairwise_distances(profiles, 'bray_curtis')
```

**Taxonomic Profile Approach (Most interpretable):**
```bash
# Kraken2 classification
for sample in *.fastq*; do
    kraken2 --db standard --threads 4 $sample > ${sample}.kraken
    bracken -d standard -i ${sample}.kraken -o ${sample}.bracken
done
```

**Similarity Interpretation:**
- **High similarity** (Bray-Curtis < 0.5) → Co-assembly beneficial
- **Moderate similarity** (Bray-Curtis 0.5-0.8) → Evaluate metadata grouping
- **Low similarity** (Bray-Curtis > 0.8) → Individual assembly preferred

### STEP 3: Metadata Variable Selection (If Co-Assembly Chosen)

#### 3A: Biological Relevance Assessment

**Rank metadata variables by expected microbial community effect:**

**High biological relevance:**
- Treatment vs control (drugs, interventions)
- Disease state (healthy vs diseased)
- Body site (gut vs oral vs skin)
- Environmental conditions (pH, temperature, salinity)
- Host species

**Medium biological relevance:**
- Time points (acute vs chronic effects)
- Geographic location (if environmentally distinct)
- Demographic variables (age, sex - context dependent)

**Low biological relevance:**
- Technical variables (batch, extraction kit)
- Random identifiers
- Variables with minimal biological basis

#### 3B: Statistical Validation (If computational assessment was performed)

**PERMANOVA Testing:**
```r
library(vegan)

# Test each metadata variable
adonis2(distance_matrix ~ treatment, data=metadata, permutations=999)
adonis2(distance_matrix ~ time_point, data=metadata, permutations=999)
adonis2(distance_matrix ~ location, data=metadata, permutations=999)

# Interpretation:
# R² > 0.15 AND p < 0.05 = Strong grouping variable
# R² > 0.05 AND p < 0.05 = Moderate grouping variable  
# R² < 0.05 OR p > 0.05 = Weak grouping variable
```

**Visualization Assessment:**
```r
# PCoA ordination
pcoa <- cmdscale(distance_matrix, eig=TRUE)
plot_data <- data.frame(PC1=pcoa$points[,1], PC2=pcoa$points[,2], metadata)

# Look for clear clustering by metadata variables
library(ggplot2)
ggplot(plot_data, aes(x=PC1, y=PC2, color=treatment)) + 
  geom_point(size=3) + theme_minimal()
```

#### 3C: Grouping Strategy Selection

**Sample Size Requirements:**
- Each group needs ≥ 3 samples (preferably ≥ 5)
- If groups have < 3 samples, consider combining categories or using similarity-based grouping

**Decision Logic:**
1. **Single strong variable** (R² > 0.15) → Use metadata-guided grouping
2. **Multiple moderate variables** → Test combinations
3. **No strong variables** (all R² < 0.05) → Use similarity-based grouping
4. **Technical confounding detected** → Address before proceeding

### STEP 4: Final Assembly Strategy Selection

## Decision Matrix

| Samples | Depth/Sample | Expected Similarity | Strong Metadata Effects | Recommended Strategy |
|---------|-------------|---------------------|------------------------|---------------------|
| < 3 | Any | Any | N/A | Individual Assembly |
| 3+ | High (>20M) | Low (BC > 0.8) | Any | Individual Assembly |
| 3+ | High (>20M) | High (BC < 0.5) | Yes (R² > 0.15) | Metadata-Guided Co-Assembly |
| 3+ | High (>20M) | High (BC < 0.5) | No (R² < 0.05) | Similarity-Based Co-Assembly |
| 3+ | Medium (5-20M) | Moderate (BC 0.5-0.8) | Yes | Metadata-Guided Co-Assembly |
| 3+ | Medium (5-20M) | Moderate (BC 0.5-0.8) | No | Similarity-Based Co-Assembly |
| 3+ | Low (<5M) | Any | Any | Pooled Co-Assembly |

*BC = Bray-Curtis distance; R² = PERMANOVA R-squared*

## Strategy Descriptions

### Individual Assembly
**When to use:**
- < 3 total samples
- High depth (>20M reads/sample) + low expected similarity
- Samples from very different environments
- Goal is sample-specific genome recovery

**Advantages:**
- Sample-specific strain resolution
- No cross-contamination between samples
- Easier to trace contigs to source samples

**Disadvantages:**
- May miss rare organisms due to depth limitations
- Less comprehensive gene catalogs
- Higher computational cost per sample

### Metadata-Guided Co-Assembly
**When to use:**
- Clear biological groupings with ≥3 samples per group
- Strong metadata effects (R² > 0.15, p < 0.05)
- Research questions focus on group differences
- Want to improve rare genome recovery within conditions

**Advantages:**
- Biologically meaningful grouping
- Improved rare organism recovery within groups
- Maintains biological interpretability
- Balanced computational efficiency

**Disadvantages:**
- Requires good metadata
- May miss cross-group contamination
- Groups must have sufficient sample sizes

### Similarity-Based Co-Assembly
**When to use:**
- High expected similarity but no clear metadata groupings
- Exploratory analysis without strong hypotheses
- Want comprehensive gene catalog from similar samples
- Metadata creates groups with < 3 samples

**Advantages:**
- Data-driven grouping
- Maximizes assembly quality for shared taxa
- No reliance on metadata quality
- Good for discovery-based research

**Disadvantages:**
- Groups may not be biologically interpretable
- Requires computational analysis for grouping
- May obscure condition-specific differences

### Pooled Co-Assembly
**When to use:**
- Low sequencing depth (< 5M reads/sample)
- Exploratory analysis of novel environments
- Interested in overall functional potential
- Limited computational resources

**Advantages:**
- Maximum depth for rare organism recovery
- Comprehensive functional gene catalogs
- Single assembly to manage
- Computationally efficient

**Disadvantages:**
- Loss of sample-specific information
- Difficult strain resolution
- May create assembly artifacts from divergent samples

## Implementation Scripts

The toolkit provides scripts for each approach:

```bash
# Distance matrix generation options
scripts/distance_sourmash.py      # Fast sketching approach
scripts/distance_kmer.py          # K-mer profile approach
scripts/distance_taxonomic.py     # Taxonomic profile approach
scripts/distance_hybrid.py        # Multi-method validation

# Assembly strategy implementation
scripts/individual_assembly.py    # Individual sample assembly
scripts/metadata_grouping.py      # Metadata-guided grouping
scripts/similarity_grouping.py    # Similarity-based grouping
scripts/pooled_assembly.py        # Single pooled assembly
```

## Distance Matrix Approach Selection

| Dataset Size | Computational Resources | Scientific Goal | Recommended Approach |
|-------------|------------------------|-----------------|---------------------|
| <20 samples | Limited | Exploratory | Sourmash sketching |
| <20 samples | Abundant | Assembly optimization | K-mer profiles |
| 20-100 samples | Limited | General | Sourmash sketching |
| 20-100 samples | Medium | Assembly optimization | K-mer profiles |
| 20-100 samples | Medium | Ecological interpretation | Taxonomic profiles |
| >100 samples | Any | Any | Sourmash sketching |
| Any size | Abundant | High confidence | Hybrid multi-method |

## Validation and Quality Control

After implementing your chosen strategy, validate results using:

### Assembly Quality Metrics
```bash
# Basic assembly statistics
assembly-stats assembly.fasta

# Genome completeness assessment  
checkm lineage_wf bins_folder checkm_output -t 8 -x fa

# Functional gene recovery
prodigal -i assembly.fasta -a proteins.faa -d genes.fna
```

### Comparative Assessment
- Compare assembly statistics between strategies tested
- Evaluate genome recovery rates
- Assess functional gene completeness
- Document computational resource usage

## Troubleshooting

### Unexpected Results
- **High depth but poor assembly**: Check for host contamination or low complexity
- **Metadata grouping doesn't match similarity**: Consider batch effects or technical confounding
- **Poor rare organism recovery**: Try increasing co-assembly group sizes

### Computational Issues
- **High memory usage**: Use smaller k-mer sizes or subsample reads
- **Long runtime**: Start with sample subsets for method testing
- **Storage limitations**: Use compressed intermediate files

### Biological Interpretation
- **Groups don't make biological sense**: Prioritize biological relevance over statistical significance
- **Conflicting metadata effects**: Consider interaction effects or hierarchical grouping
- **Technical batch effects**: Use batch correction methods before grouping

## Best Practices

1. **Start Simple**: Begin with biological knowledge-based decisions when possible
2. **Validate Computationally**: Use distance matrix analysis to confirm biological expectations
3. **Document Decisions**: Record rationale for grouping strategy selection
4. **Compare Approaches**: Test multiple strategies on pilot data when feasible
5. **Consider Resources**: Balance scientific rigor with computational constraints

## Integration with Existing Workflows

This decision framework integrates with:
- **Quality Control**: Use after initial QC but before assembly
- **Scientific Validity Checks**: Combine with bias correction and compositional analysis
- **Assembly Pipelines**: Provides grouping strategies for existing assemblers
- **Downstream Analysis**: Maintains sample relationships for post-assembly analysis

---

## Quick Reference Commands

### Minimal Assessment (5 minutes)
```bash
# Count samples and estimate depth
find . -name "*.fastq*" | wc -l
ls -lh *.fastq* | head -5

# Make decision based on biological knowledge
# < 3 samples → Individual assembly
# Different environments → Individual assembly  
# Similar environments → Co-assembly
```

### Computational Assessment (30-60 minutes)
```bash
# Generate distance matrix (choose one approach)
python scripts/distance_sourmash.py --input-dir . --output distances.csv

# Analyze metadata effects
python scripts/analyze_metadata.py --distances distances.csv --metadata metadata.csv

# Generate grouping recommendations
python scripts/recommend_strategy.py --distances distances.csv --metadata metadata.csv
```

### Full Analysis Pipeline
```bash
# Comprehensive analysis with all validation steps
python assembly_strategy_optimizer.py \
  --samples samples.txt \
  --metadata metadata.csv \
  --output results/ \
  --distance-method kmer \
  --validate-groupings \
  --generate-report
```