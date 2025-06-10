# Assembly Strategy Decision Tree Guide

A systematic framework for selecting optimal assembly strategies for metagenomic datasets.

## Overview

This guide provides a structured decision-making process to help researchers choose between different assembly approaches based on their data characteristics, research goals, and available resources. The framework considers four main assembly strategies:

1. **Individual Assembly**: Each sample assembled separately
2. **Metadata-Guided Co-Assembly**: Samples grouped by biological metadata variables
3. **Similarity-Based Co-Assembly**: Samples grouped by sequence similarity
4. **Pooled Co-Assembly**: All samples assembled together

## Decision Tree Framework

### STEP 1: Data Type and Preparation Assessment

**Determine your data characteristics (2-3 minutes):**

1. **What type of metagenomic preparation was used?**
   - **Standard total DNA** → Bacterial/archaeal dominant communities
   - **VLP (virus-like particle) enriched** → Viral-enriched samples
   - **Host-associated** (gut, skin, oral) → Variable host contamination
   - **Environmental** (soil, water, sediment) → High complexity expected
   - **Other specialized prep** → May need custom approach

2. **Expected community composition:**
   - **Mixed prokaryotic** → Standard workflow
   - **Viral-dominant** → Modified workflow (see VLP-specific guidance)
   - **Host-contaminated** → Emphasize decontamination steps
   - **Unknown/exploratory** → Use conservative, robust approaches

3. **PCR amplification bias risk assessment:**
   - **High risk**: VLP preparations, low biomass samples, environmental DNA, ancient DNA
   - **Medium risk**: Host-associated samples with low microbial load
   - **Low risk**: High biomass samples (fecal, soil), minimal PCR cycles used
   - **Unknown**: Check FastQC duplication rates and GC distribution

### STEP 2: Data Quality and Preprocessing

**Essential preprocessing steps (30-90 minutes depending on dataset size):**

#### 2A: Quality Assessment and Filtering
```bash
# Check raw read quality
fastqc *.fastq* -o qc_reports/

# Quality trimming (if needed)
fastp -i sample_R1.fastq -I sample_R2.fastq \
  -o sample_R1_clean.fastq -O sample_R2_clean.fastq \
  --detect_adapter_for_pe --cut_tail --cut_tail_window_size 4 --cut_tail_mean_quality 20
```

#### 2B: Host Contamination Removal (if applicable)
**For host-associated samples:**
```bash
# Remove host sequences
bowtie2 --very-sensitive-local -x host_genome -1 sample_R1.fastq -2 sample_R2.fastq \
  --un-conc sample_dehost.fastq

# For unknown host contamination, use k-mer based methods
kraken2 --db standard sample.fastq --unclassified-out sample_microbial.fastq
```

**For VLP/viral-enriched samples:**
- Host removal usually minimal or unnecessary
- Focus on removing residual bacterial contamination if needed

#### 2C: Sequencing Depth Normalization and Bias Assessment

**First, assess for PCR amplification bias:**
```bash
# Check for potential PCR bias indicators
fastqc *.fastq* -o qc_reports/
# Look for: unusual GC distribution, high duplication rates, uneven coverage

# For VLP/viral data, PCR bias is common due to low starting material
```

**Important PCR Bias Considerations:**
- **If PCR bias is suspected or known** (e.g., VLP preparations, low biomass samples):
  - ⚠️ **Avoid abundance-based analyses** (relative abundance comparisons unreliable)
  - ⚠️ **Use presence/absence methods** instead of abundance-weighted metrics
  - ⚠️ **Focus on diversity rather than composition**
  - ⚠️ **Be cautious with quantitative conclusions**

**Choose normalization approach:**
```bash
# Option 1: Subsampling to minimum depth (recommended for PCR bias)
seqtk sample -s 100 sample.fastq 1000000 > sample_subsample.fastq

# Option 2: Scaling (only if no suspected PCR bias)
python core/normalization.py --method scaling --input sample_counts.csv --output normalized_counts.csv

# Option 3: For PCR-biased data, use presence/absence normalization
python core/normalization.py --method binary --input sample_counts.csv --output binary_counts.csv
```

#### 2D: Final Quality Check
```bash
# Basic file inventory after preprocessing
find . -name "*_clean.fastq*" | wc -l
ls -lh *_clean.fastq* | head -10

# Verify read counts are reasonable
for file in *_clean.fastq*; do
  echo "$file: $(wc -l < $file | awk '{print $1/4}')"
done
```

### STEP 3: Sample Count and Depth Assessment

**Decision Points (adjust based on data type):**

#### For Standard Metagenomic Data:
1. **Sample Count Assessment**
   - **< 3 samples** → **Individual Assembly Only**
   - **3-10 samples** → Limited grouping options available
   - **> 10 samples** → Full range of strategies available

2. **Sequencing Depth per Sample (after preprocessing)**
   - **> 20M reads/sample** → Individual assembly viable
   - **5-20M reads/sample** → Evaluate co-assembly benefits
   - **< 5M reads/sample** → Co-assembly likely necessary

#### For VLP/Viral-Enriched Data:
1. **Sample Count Assessment**
   - **< 5 samples** → Individual assembly often sufficient (lower complexity)
   - **5-15 samples** → Consider metadata-guided grouping
   - **> 15 samples** → Full range of strategies available

2. **Sequencing Depth per Sample**
   - **> 10M reads/sample** → Individual assembly often viable (viral genomes shorter)
   - **2-10M reads/sample** → Evaluate co-assembly for rare virus recovery
   - **< 2M reads/sample** → Co-assembly recommended

3. **Read Type Considerations (both data types)**
   - **Single-end reads** → Co-assembly preferred (assembly challenges)
   - **Paired-end reads** → All strategies viable
   - **Long reads (ONT/PacBio)** → Different workflow entirely
   - **Read length < 100bp** → Co-assembly recommended

### STEP 4: Community Structure Assessment

**Choose your assessment approach based on available resources:**

#### Option A: Biological Knowledge-Based (2 minutes)
Answer based on your experimental design:

1. **Are samples from similar environments?**
   - YES (same body site, similar soil types) → Consider co-assembly
   - NO (gut vs soil, different hosts) → Individual assembly

2. **Do you expect shared species/organisms between samples?**
   - YES → Co-assembly beneficial
   - NO → Individual assembly
   - UNSURE → Default to co-assembly (safer for discovery)

**Data type considerations:**
- **VLP data**: Higher chance of shared viral families between similar environments
- **Host-associated**: Strong host effect usually means shared taxa
- **Environmental**: Geographic/temporal proximity increases sharing likelihood

#### Option B: Rapid Computational Assessment (30-60 minutes)
**IMPORTANT: Use preprocessed, normalized data from Step 2**

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

# Choose distance metric based on PCR bias status
if pcr_bias_suspected:
    distance_matrix = calc.calculate_pairwise_distances(profiles, 'jaccard')  # Presence/absence
else:
    distance_matrix = calc.calculate_pairwise_distances(profiles, 'bray_curtis')  # Abundance-weighted
```

**Taxonomic Profile Approach (Most interpretable, but abundance-sensitive):**
```bash
# Kraken2 classification
for sample in *.fastq*; do
    kraken2 --db standard --threads 4 $sample > ${sample}.kraken
    bracken -d standard -i ${sample}.kraken -o ${sample}.bracken
done

# ⚠️ WARNING: If PCR bias suspected, taxonomic abundance profiles may be unreliable
# Consider using presence/absence of taxa instead of relative abundances
```

**Similarity Interpretation (adjust thresholds based on data type):**

**For Standard Metagenomic Data:**
- **High similarity** (Bray-Curtis < 0.5) → Co-assembly beneficial
- **Moderate similarity** (Bray-Curtis 0.5-0.8) → Evaluate metadata grouping
- **Low similarity** (Bray-Curtis > 0.8) → Individual assembly preferred

**For VLP/Viral-Enriched Data:**
- **High similarity** (Bray-Curtis < 0.3) → Co-assembly beneficial (viruses more variable)
- **Moderate similarity** (Bray-Curtis 0.3-0.7) → Evaluate metadata grouping
- **Low similarity** (Bray-Curtis > 0.7) → Individual assembly preferred

### STEP 5: Metadata Variable Selection (If Co-Assembly Chosen)

#### 5A: Biological Relevance Assessment

**Rank metadata variables by expected community effect (adjust by data type):**

**For Standard Metagenomic Data:**
- **High relevance**: Treatment vs control, disease state, body site, environmental conditions, host species
- **Medium relevance**: Time points, geographic location, demographic variables
- **Low relevance**: Technical variables, random identifiers

**For VLP/Viral-Enriched Data:**
- **High relevance**: Host species, body site, disease state, antiviral treatments, immune status
- **Medium relevance**: Time points (viral dynamics faster), geographic location, environmental stressors
- **Low relevance**: Most demographic variables (unless immune-related), dietary factors

**Additional VLP considerations:**
- Seasonal effects may be stronger for environmental viruses
- Treatment effects may be more pronounced (direct antiviral action)
- Host immunity variables become critical

#### 5B: Statistical Validation (If computational assessment was performed)

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

#### 5C: Grouping Strategy Selection

**Sample Size Requirements (adjust by data type):**
- **Standard metagenomic**: Each group needs ≥ 3 samples (preferably ≥ 5)
- **VLP/viral-enriched**: Each group needs ≥ 2 samples (lower complexity allows smaller groups)
- If groups are too small, consider combining categories or using similarity-based grouping

**Decision Logic:**
1. **Single strong variable** (R² > 0.15 standard, R² > 0.10 for VLP) → Use metadata-guided grouping
2. **Multiple moderate variables** → Test combinations
3. **No strong variables** (all R² < 0.05) → Use similarity-based grouping
4. **Technical confounding detected** → Address before proceeding

### STEP 6: Final Assembly Strategy Selection

## Decision Matrices

### For Standard Metagenomic Data

| Samples | Depth/Sample | Expected Similarity | Strong Metadata Effects | Recommended Strategy |
|---------|-------------|---------------------|------------------------|---------------------|
| < 3 | Any | Any | N/A | Individual Assembly |
| 3+ | High (>20M) | Low (BC > 0.8) | Any | Individual Assembly |
| 3+ | High (>20M) | High (BC < 0.5) | Yes (R² > 0.15) | Metadata-Guided Co-Assembly |
| 3+ | High (>20M) | High (BC < 0.5) | No (R² < 0.05) | Similarity-Based Co-Assembly |
| 3+ | Medium (5-20M) | Moderate (BC 0.5-0.8) | Yes (R² > 0.15) | Metadata-Guided Co-Assembly |
| 3+ | Medium (5-20M) | Moderate (BC 0.5-0.8) | No (R² < 0.05) | Similarity-Based Co-Assembly |
| 3+ | Low (<5M) | Any | Any | Pooled Co-Assembly |

### For VLP/Viral-Enriched Data

| Samples | Depth/Sample | Expected Similarity | Strong Metadata Effects | Recommended Strategy |
|---------|-------------|---------------------|------------------------|---------------------|
| < 2 | Any | Any | N/A | Individual Assembly |
| 2-4 | High (>10M) | Low (BC > 0.7) | Any | Individual Assembly |
| 2-4 | High (>10M) | High (BC < 0.3) | Yes (R² > 0.10) | Metadata-Guided Co-Assembly |
| 5+ | High (>10M) | High (BC < 0.3) | Yes (R² > 0.10) | Metadata-Guided Co-Assembly |
| 5+ | High (>10M) | High (BC < 0.3) | No (R² < 0.05) | Similarity-Based Co-Assembly |
| 5+ | Medium (2-10M) | Moderate (BC 0.3-0.7) | Yes (R² > 0.10) | Metadata-Guided Co-Assembly |
| 5+ | Medium (2-10M) | Moderate (BC 0.3-0.7) | No (R² < 0.05) | Similarity-Based Co-Assembly |
| Any | Low (<2M) | Any | Any | Pooled Co-Assembly |

*BC = Bray-Curtis distance; R² = PERMANOVA R-squared*

**Key differences for VLP data:**
- Lower depth thresholds (viral genomes are shorter)
- Stricter similarity thresholds (viral communities more variable)
- Lower R² thresholds for metadata effects (smaller effect sizes expected)
- Smaller minimum group sizes acceptable

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

After implementing your chosen strategy, validate results using data-type appropriate methods:

### Standard Metagenomic Data Quality Metrics
```bash
# Basic assembly statistics
assembly-stats assembly.fasta

# Bacterial/archaeal genome completeness assessment
checkm lineage_wf bins_folder checkm_output -t 8 -x fa

# Functional gene recovery
prodigal -i assembly.fasta -a proteins.faa -d genes.fna -p meta

# Taxonomic annotation
kraken2 --db standard assembly.fasta > assembly_taxonomy.txt
```

### VLP/Viral-Enriched Data Quality Metrics
```bash
# Basic assembly statistics (focus on shorter contigs)
assembly-stats assembly.fasta

# Viral genome completeness assessment
checkv end_to_end assembly.fasta checkv_output

# Viral gene prediction and annotation
prodigal -i assembly.fasta -a viral_proteins.faa -d viral_genes.fna -p meta
hmmsearch --domtblout viral_hallmarks.out viral_hallmark_genes.hmm viral_proteins.faa

# Viral taxonomy (if available)
# Note: Viral databases are less complete than bacterial
blastn -query assembly.fasta -db viral_refseq -outfmt 6 -max_target_seqs 5
```

### Quality Thresholds by Data Type

**Standard Metagenomic:**
- Minimum contig length: 1000 bp
- Genome completeness: >80% (CheckM)
- Contamination: <10% (CheckM)
- N50: >10 kb (for complex communities)

**VLP/Viral-Enriched:**
- Minimum contig length: 500 bp (viral genomes shorter)
- Genome completeness: >70% (CheckV, when applicable)
- Contamination: <5% host sequences
- N50: >2 kb (viral genomes naturally shorter)

### Comparative Assessment
- Compare assembly statistics between strategies tested
- Evaluate genome/viral recovery rates
- Assess functional gene completeness
- Document computational resource usage
- **For VLP**: Focus on recovery of viral hallmark genes and diversity metrics

## Troubleshooting

### PCR Amplification Bias Issues
- **Suspected PCR bias in data**:
  - ✓ **Switch to presence/absence metrics** (Jaccard instead of Bray-Curtis)
  - ✓ **Avoid quantitative abundance conclusions**
  - ✓ **Focus on diversity and presence patterns rather than relative abundances**
  - ✓ **Use binary normalization methods**
  - ✓ **Be conservative in statistical interpretations**

- **VLP/viral data showing unusual patterns**:
  - ✓ **Expected**: Higher variability, lower similarity between samples
  - ✓ **Check duplication rates** in FastQC (>50% may indicate PCR bias)
  - ✓ **Consider individual assembly** even with moderate sample numbers
  - ✓ **Use presence/absence for community comparisons**

### Unexpected Results
- **High depth but poor assembly**: Check for host contamination or low complexity
- **Metadata grouping doesn't match similarity**: Consider batch effects, technical confounding, or PCR bias
- **Poor rare organism recovery**: Try increasing co-assembly group sizes (but be cautious if PCR bias present)

### Computational Issues
- **High memory usage**: Use smaller k-mer sizes or subsample reads
- **Long runtime**: Start with sample subsets for method testing
- **Storage limitations**: Use compressed intermediate files

### Biological Interpretation
- **Groups don't make biological sense**: Prioritize biological relevance over statistical significance
- **Conflicting metadata effects**: Consider interaction effects, hierarchical grouping, or PCR bias artifacts
- **Technical batch effects**: Use batch correction methods before grouping, check for PCR bias

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
# Step 1: Determine data type and count samples
echo "Data type: [standard|VLP|host-associated|environmental]"
find . -name "*.fastq*" | wc -l
ls -lh *.fastq* | head -5

# Step 2: Make decision based on biological knowledge and data type
# Standard metagenomic: < 3 samples → Individual assembly
# VLP data: < 2 samples → Individual assembly  
# Different environments → Individual assembly  
# Similar environments → Co-assembly
```

### Standard Metagenomic Workflow (60-90 minutes)
```bash
# Preprocessing with host removal
python preprocess_samples.py --input-dir . --remove-host --host-db human_genome

# Generate distance matrix
python scripts/distance_kmer.py --input-dir preprocessed/ --output distances.csv

# Analyze metadata effects
python scripts/analyze_metadata.py --distances distances.csv --metadata metadata.csv

# Generate grouping recommendations  
python scripts/recommend_strategy.py --distances distances.csv --metadata metadata.csv
```

### VLP/Viral-Enriched Workflow (30-60 minutes)
```bash
# Lighter preprocessing (minimal host removal)
python preprocess_samples.py --input-dir . --light-filtering --viral-optimized

# ⚠️ IMPORTANT: For VLP data, assess PCR bias first
# VLP preparations often have PCR bias due to low starting material

# Generate distance matrix with presence/absence focus (if PCR bias suspected)
python scripts/distance_sourmash.py --input-dir preprocessed/ --output distances.csv --viral-mode --presence-absence

# Analyze metadata effects with adjusted thresholds and bias-aware methods
python scripts/analyze_metadata.py --distances distances.csv --metadata metadata.csv --viral-thresholds --pcr-bias-mode

# Generate grouping recommendations
python scripts/recommend_strategy.py --distances distances.csv --metadata metadata.csv --data-type viral --pcr-bias-aware
```

### Full Analysis Pipeline
```bash
# Comprehensive analysis with all validation steps
python assembly_strategy_optimizer.py \
  --samples samples.txt \
  --metadata metadata.csv \
  --output results/ \
  --data-type [standard|viral] \
  --distance-method kmer \
  --validate-groupings \
  --generate-report
```