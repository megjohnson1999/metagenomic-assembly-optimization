"""K-mer based distance calculation for metagenomic samples."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import multiprocessing as mp
from functools import partial
import logging
from collections import Counter
import gzip
from itertools import product

logger = logging.getLogger(__name__)


class KmerDistanceCalculator:
    """Calculate k-mer based distances between metagenomic samples.
    
    This class provides methods to compute k-mer profiles from FASTQ files
    and calculate various distance metrics between samples.
    """
    
    def __init__(self, k: int = 4, n_processes: int = None):
        """Initialize the k-mer distance calculator.
        
        Args:
            k: K-mer size (default: 4)
            n_processes: Number of processes for parallel computation
                        (default: number of CPU cores)
        """
        self.k = k
        self.n_processes = n_processes or mp.cpu_count()
        self.kmers = self._generate_all_kmers()
        
    def _generate_all_kmers(self) -> List[str]:
        """Generate all possible k-mers of size k."""
        bases = ['A', 'C', 'G', 'T']
        return [''.join(kmer) for kmer in product(bases, repeat=self.k)]
    
    def _reverse_complement(self, seq: str) -> str:
        """Generate reverse complement of a DNA sequence."""
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        return ''.join(complement.get(base, 'N') for base in seq[::-1])
    
    def _canonical_kmer(self, kmer: str) -> str:
        """Return the canonical k-mer (lexicographically smaller of k-mer and its reverse complement)."""
        rev_comp = self._reverse_complement(kmer)
        return min(kmer, rev_comp)
    
    def _count_kmers_in_read(self, sequence: str) -> Counter:
        """Count k-mers in a single sequence."""
        kmer_counts = Counter()
        seq_upper = sequence.upper()
        
        for i in range(len(seq_upper) - self.k + 1):
            kmer = seq_upper[i:i + self.k]
            if 'N' not in kmer:  # Skip k-mers with ambiguous bases
                canonical = self._canonical_kmer(kmer)
                kmer_counts[canonical] += 1
                
        return kmer_counts
    
    def _process_fastq_file(self, filepath: Path, max_reads: Optional[int] = None) -> Dict[str, float]:
        """Process a single FASTQ file and return k-mer frequencies.
        
        Args:
            filepath: Path to FASTQ file (can be gzipped)
            max_reads: Maximum number of reads to process (None for all)
            
        Returns:
            Dictionary of k-mer frequencies
        """
        kmer_counts = Counter()
        total_kmers = 0
        reads_processed = 0
        
        # Determine if file is gzipped
        open_func = gzip.open if str(filepath).endswith('.gz') else open
        
        try:
            with open_func(filepath, 'rt') as f:
                while True:
                    # Read four lines (FASTQ format)
                    header = f.readline().strip()
                    if not header:
                        break
                        
                    sequence = f.readline().strip()
                    plus = f.readline().strip()
                    quality = f.readline().strip()
                    
                    if not all([header.startswith('@'), plus.startswith('+'), sequence, quality]):
                        logger.warning(f"Malformed FASTQ entry at read {reads_processed}")
                        continue
                    
                    # Count k-mers in this read
                    read_kmers = self._count_kmers_in_read(sequence)
                    kmer_counts.update(read_kmers)
                    total_kmers += sum(read_kmers.values())
                    
                    reads_processed += 1
                    if max_reads and reads_processed >= max_reads:
                        break
                        
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            raise
        
        # Convert to frequencies
        kmer_freqs = {}
        if total_kmers > 0:
            for kmer in self.kmers:
                canonical = self._canonical_kmer(kmer)
                kmer_freqs[canonical] = kmer_counts.get(canonical, 0) / total_kmers
        
        logger.info(f"Processed {reads_processed} reads from {filepath}")
        return kmer_freqs
    
    def calculate_kmer_profiles(self, sample_files: Dict[str, Union[Path, Tuple[Path, Path]]], 
                               max_reads_per_sample: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """Calculate k-mer profiles for multiple samples.
        
        Args:
            sample_files: Dictionary mapping sample names to file paths
                         (single file or tuple of paired files)
            max_reads_per_sample: Maximum reads to process per sample
            
        Returns:
            Dictionary mapping sample names to k-mer frequency dictionaries
        """
        profiles = {}
        
        for sample_name, files in sample_files.items():
            logger.info(f"Processing sample: {sample_name}")
            
            if isinstance(files, tuple):
                # Paired-end reads
                profile1 = self._process_fastq_file(files[0], max_reads_per_sample)
                profile2 = self._process_fastq_file(files[1], max_reads_per_sample)
                
                # Combine profiles (average frequencies)
                combined_profile = {}
                all_kmers = set(profile1.keys()) | set(profile2.keys())
                for kmer in all_kmers:
                    combined_profile[kmer] = (profile1.get(kmer, 0) + profile2.get(kmer, 0)) / 2
                
                profiles[sample_name] = combined_profile
            else:
                # Single-end reads
                profiles[sample_name] = self._process_fastq_file(files, max_reads_per_sample)
        
        return profiles
    
    def calculate_distance_matrix(self, profiles: Dict[str, Dict[str, float]], 
                                 metric: str = 'braycurtis') -> np.ndarray:
        """Calculate pairwise distances between samples.
        
        Args:
            profiles: Dictionary of k-mer profiles
            metric: Distance metric ('braycurtis', 'euclidean', 'cosine', 'jaccard')
            
        Returns:
            Distance matrix as numpy array
        """
        sample_names = list(profiles.keys())
        n_samples = len(sample_names)
        distance_matrix = np.zeros((n_samples, n_samples))
        
        # Convert profiles to matrix format
        all_kmers = sorted(set().union(*[set(p.keys()) for p in profiles.values()]))
        profile_matrix = np.zeros((n_samples, len(all_kmers)))
        
        for i, sample in enumerate(sample_names):
            for j, kmer in enumerate(all_kmers):
                profile_matrix[i, j] = profiles[sample].get(kmer, 0)
        
        # Calculate distances
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if metric == 'braycurtis':
                    dist = self._braycurtis_distance(profile_matrix[i], profile_matrix[j])
                elif metric == 'euclidean':
                    dist = np.linalg.norm(profile_matrix[i] - profile_matrix[j])
                elif metric == 'cosine':
                    dist = self._cosine_distance(profile_matrix[i], profile_matrix[j])
                elif metric == 'jaccard':
                    dist = self._jaccard_distance(profile_matrix[i], profile_matrix[j])
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        return distance_matrix, sample_names
    
    def _braycurtis_distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """Calculate Bray-Curtis distance between two vectors."""
        numerator = np.sum(np.abs(u - v))
        denominator = np.sum(u) + np.sum(v)
        return numerator / denominator if denominator > 0 else 0
    
    def _cosine_distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """Calculate cosine distance between two vectors."""
        dot_product = np.dot(u, v)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        
        if norm_u == 0 or norm_v == 0:
            return 1.0
        
        cosine_similarity = dot_product / (norm_u * norm_v)
        return 1 - cosine_similarity
    
    def _jaccard_distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """Calculate Jaccard distance between two vectors."""
        u_binary = u > 0
        v_binary = v > 0
        
        intersection = np.sum(u_binary & v_binary)
        union = np.sum(u_binary | v_binary)
        
        return 1 - (intersection / union) if union > 0 else 0