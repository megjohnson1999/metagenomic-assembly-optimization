#!/usr/bin/env python3
"""
Create synthetic gut microbiome FASTQ files for testing the assembly optimization toolkit.

This creates realistic test data with different microbial compositions that can be used
to test k-mer distance calculations and sample grouping.
"""

import os
import random
import gzip
from pathlib import Path
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_synthetic_reads(output_file, n_reads=10000, read_length=150, organisms=None):
    """Create synthetic FASTQ reads based on organism signatures."""
    
    # Default organism signatures (simplified k-mer patterns)
    if organisms is None:
        organisms = {
            'Bacteroides': {'GC_content': 0.43, 'signature': 'ATCGATCG', 'abundance': 0.4},
            'Firmicutes': {'GC_content': 0.35, 'signature': 'GCTAGCTA', 'abundance': 0.3}, 
            'Proteobacteria': {'GC_content': 0.52, 'signature': 'TTAAGGCC', 'abundance': 0.2},
            'Actinobacteria': {'GC_content': 0.69, 'signature': 'GGCCGGCC', 'abundance': 0.1}
        }
    
    # Generate reads
    reads = []
    for i in range(n_reads):
        # Select organism based on abundance
        org_choice = random.choices(
            list(organisms.keys()), 
            weights=[org['abundance'] for org in organisms.values()]
        )[0]
        
        org_info = organisms[org_choice]
        
        # Generate read with organism signature
        read_seq = generate_read_with_signature(
            read_length, 
            org_info['GC_content'],
            org_info['signature']
        )
        
        # Create FASTQ entry
        read_id = f"@read_{i+1}_{org_choice}"
        quality = "I" * read_length  # High quality scores
        
        reads.append(f"{read_id}\n{read_seq}\n+\n{quality}\n")
    
    # Write to file
    if output_file.endswith('.gz'):
        with gzip.open(output_file, 'wt') as f:
            f.writelines(reads)
    else:
        with open(output_file, 'w') as f:
            f.writelines(reads)
    
    logger.info(f"Created {output_file} with {n_reads} reads")

def generate_read_with_signature(length, gc_content, signature, signature_freq=0.1):
    """Generate a DNA read with specific GC content and organism signature."""
    read = []
    signature_positions = set(random.sample(range(0, length-len(signature)), 
                                          max(1, int(length * signature_freq / len(signature)))))
    
    for i in range(length):
        # Insert signature at specific positions
        if i in signature_positions and i + len(signature) <= length:
            read.extend(list(signature))
            i += len(signature) - 1
        else:
            # Generate base based on GC content
            if random.random() < gc_content / 2:
                base = random.choice(['G', 'C'])
            else:
                base = random.choice(['A', 'T'])
            read.append(base)
    
    return ''.join(read[:length])

def create_gut_microbiome_samples():
    """Create diverse gut microbiome samples with different compositions."""
    
    # Create output directory
    os.makedirs("test_fastq_files", exist_ok=True)
    
    # Define different sample types
    sample_configs = {
        # Healthy samples (diverse microbiome)
        'healthy_01': {
            'Bacteroides': {'GC_content': 0.43, 'signature': 'ATCGATCG', 'abundance': 0.35},
            'Firmicutes': {'GC_content': 0.35, 'signature': 'GCTAGCTA', 'abundance': 0.40}, 
            'Proteobacteria': {'GC_content': 0.52, 'signature': 'TTAAGGCC', 'abundance': 0.15},
            'Actinobacteria': {'GC_content': 0.69, 'signature': 'GGCCGGCC', 'abundance': 0.10}
        },
        'healthy_02': {
            'Bacteroides': {'GC_content': 0.43, 'signature': 'ATCGATCG', 'abundance': 0.30},
            'Firmicutes': {'GC_content': 0.35, 'signature': 'GCTAGCTA', 'abundance': 0.45}, 
            'Proteobacteria': {'GC_content': 0.52, 'signature': 'TTAAGGCC', 'abundance': 0.15},
            'Actinobacteria': {'GC_content': 0.69, 'signature': 'GGCCGGCC', 'abundance': 0.10}
        },
        'healthy_03': {
            'Bacteroides': {'GC_content': 0.43, 'signature': 'ATCGATCG', 'abundance': 0.40},
            'Firmicutes': {'GC_content': 0.35, 'signature': 'GCTAGCTA', 'abundance': 0.35}, 
            'Proteobacteria': {'GC_content': 0.52, 'signature': 'TTAAGGCC', 'abundance': 0.15},
            'Actinobacteria': {'GC_content': 0.69, 'signature': 'GGCCGGCC', 'abundance': 0.10}
        },
        
        # Dysbiotic samples (imbalanced microbiome)
        'dysbiotic_01': {
            'Bacteroides': {'GC_content': 0.43, 'signature': 'ATCGATCG', 'abundance': 0.65},
            'Firmicutes': {'GC_content': 0.35, 'signature': 'GCTAGCTA', 'abundance': 0.15}, 
            'Proteobacteria': {'GC_content': 0.52, 'signature': 'TTAAGGCC', 'abundance': 0.15},
            'Actinobacteria': {'GC_content': 0.69, 'signature': 'GGCCGGCC', 'abundance': 0.05}
        },
        'dysbiotic_02': {
            'Bacteroides': {'GC_content': 0.43, 'signature': 'ATCGATCG', 'abundance': 0.70},
            'Firmicutes': {'GC_content': 0.35, 'signature': 'GCTAGCTA', 'abundance': 0.10}, 
            'Proteobacteria': {'GC_content': 0.52, 'signature': 'TTAAGGCC', 'abundance': 0.15},
            'Actinobacteria': {'GC_content': 0.69, 'signature': 'GGCCGGCC', 'abundance': 0.05}
        },
        
        # High-fiber diet samples  
        'high_fiber_01': {
            'Bacteroides': {'GC_content': 0.43, 'signature': 'ATCGATCG', 'abundance': 0.25},
            'Firmicutes': {'GC_content': 0.35, 'signature': 'GCTAGCTA', 'abundance': 0.55}, 
            'Proteobacteria': {'GC_content': 0.52, 'signature': 'TTAAGGCC', 'abundance': 0.10},
            'Actinobacteria': {'GC_content': 0.69, 'signature': 'GGCCGGCC', 'abundance': 0.10}
        },
        'high_fiber_02': {
            'Bacteroides': {'GC_content': 0.43, 'signature': 'ATCGATCG', 'abundance': 0.20},
            'Firmicutes': {'GC_content': 0.35, 'signature': 'GCTAGCTA', 'abundance': 0.60}, 
            'Proteobacteria': {'GC_content': 0.52, 'signature': 'TTAAGGCC', 'abundance': 0.10},
            'Actinobacteria': {'GC_content': 0.69, 'signature': 'GGCCGGCC', 'abundance': 0.10}
        }
    }
    
    # Create FASTQ files for each sample
    sample_manifest = []
    
    for sample_id, organisms in sample_configs.items():
        fastq_file = f"test_fastq_files/{sample_id}.fastq.gz"
        
        # Create reads (smaller number for testing)
        create_synthetic_reads(
            fastq_file, 
            n_reads=5000,  # Small for quick testing
            organisms=organisms
        )
        
        # Add to manifest
        sample_manifest.append({
            'sample_id': sample_id,
            'fastq_file': fastq_file,
            'sample_type': sample_id.split('_')[0],
            'sequencing_depth': 5000,
            'description': f"Synthetic gut microbiome sample - {sample_id.replace('_', ' ')}"
        })
    
    return sample_manifest

def create_metadata(sample_manifest):
    """Create sample metadata file."""
    
    # Add additional metadata
    for sample in sample_manifest:
        sample_type = sample['sample_type']
        
        if sample_type == 'healthy':
            sample.update({
                'condition': 'healthy',
                'diet': 'balanced',
                'age_group': random.choice(['young_adult', 'middle_age']),
                'sex': random.choice(['M', 'F']),
                'bmi_category': 'normal',
                'batch': random.choice(['batch_1', 'batch_2'])
            })
        elif sample_type == 'dysbiotic':
            sample.update({
                'condition': 'dysbiotic',
                'diet': 'western',
                'age_group': random.choice(['middle_age', 'elderly']),
                'sex': random.choice(['M', 'F']),
                'bmi_category': random.choice(['overweight', 'obese']),
                'batch': random.choice(['batch_1', 'batch_2'])
            })
        elif sample_type == 'high':  # high_fiber
            sample.update({
                'condition': 'healthy',
                'diet': 'high_fiber',
                'age_group': random.choice(['young_adult', 'middle_age']),
                'sex': random.choice(['M', 'F']),
                'bmi_category': 'normal',
                'batch': random.choice(['batch_1', 'batch_2'])
            })
    
    # Save metadata
    metadata_df = pd.DataFrame(sample_manifest)
    metadata_df.to_csv("test_sample_metadata.csv", index=False)
    
    logger.info(f"Created metadata for {len(sample_manifest)} samples")
    return metadata_df

def main():
    """Create test dataset."""
    logger.info("Creating synthetic gut microbiome test dataset...")
    
    # Create FASTQ files
    sample_manifest = create_gut_microbiome_samples()
    
    # Create metadata
    metadata_df = create_metadata(sample_manifest)
    
    logger.info("Test dataset created successfully!")
    logger.info(f"Files created:")
    logger.info(f"- test_fastq_files/ (directory with {len(sample_manifest)} FASTQ files)")
    logger.info(f"- test_sample_metadata.csv")
    
    print("\nDataset summary:")
    print(metadata_df.groupby(['condition', 'diet']).size())
    
    print("\nNext steps:")
    print("1. Run: python test_assembly_optimization.py")
    print("2. This will test k-mer distances and sample grouping")

if __name__ == "__main__":
    main()