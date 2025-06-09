"""Normalization methods for k-mer count data to reduce sequencing biases."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union, List
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class KmerNormalizer:
    """Normalize k-mer count matrices to reduce technical biases.
    
    This class implements multiple normalization strategies commonly used
    in metagenomics and RNA-seq analysis, adapted for k-mer data.
    """
    
    def __init__(self):
        """Initialize the k-mer normalizer."""
        self.normalization_factors = None
        self.method_used = None
        
    def normalize(self, kmer_counts: Union[np.ndarray, pd.DataFrame], 
                  method: str = 'css',
                  reference_sample: Optional[int] = None) -> Union[np.ndarray, pd.DataFrame]:
        """Apply normalization to k-mer count matrix.
        
        Args:
            kmer_counts: Matrix of k-mer counts (samples x k-mers)
            method: Normalization method ('css', 'tmm', 'rle', 'tss', 'clr')
            reference_sample: Reference sample index for TMM/RLE (None for auto-select)
            
        Returns:
            Normalized k-mer count matrix
        """
        # Convert to numpy array for processing
        if isinstance(kmer_counts, pd.DataFrame):
            counts_array = kmer_counts.values
            return_df = True
            df_index = kmer_counts.index
            df_columns = kmer_counts.columns
        else:
            counts_array = kmer_counts.copy()
            return_df = False
            
        # Ensure float type
        counts_array = counts_array.astype(float)
        
        # Apply normalization
        if method == 'css':
            normalized = self._css_normalization(counts_array)
        elif method == 'tmm':
            normalized = self._tmm_normalization(counts_array, reference_sample)
        elif method == 'rle':
            normalized = self._rle_normalization(counts_array, reference_sample)
        elif method == 'tss':
            normalized = self._tss_normalization(counts_array)
        elif method == 'clr':
            normalized = self._clr_normalization(counts_array)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        self.method_used = method
        
        # Return in original format
        if return_df:
            return pd.DataFrame(normalized, index=df_index, columns=df_columns)
        return normalized
    
    def _css_normalization(self, counts: np.ndarray) -> np.ndarray:
        """Cumulative Sum Scaling normalization (metagenomeSeq-inspired).
        
        This method finds a percentile in each sample's count distribution
        and scales counts so that the sum up to this percentile is equal
        across samples.
        
        Args:
            counts: Raw count matrix (samples x features)
            
        Returns:
            CSS-normalized counts
        """
        n_samples, n_features = counts.shape
        normalized = np.zeros_like(counts, dtype=float)
        
        # Calculate normalization factors
        scaling_factors = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Get non-zero counts for this sample
            sample_counts = counts[i, :]
            non_zero_counts = sample_counts[sample_counts > 0]
            
            if len(non_zero_counts) == 0:
                scaling_factors[i] = 1.0
                continue
                
            # Sort counts
            sorted_counts = np.sort(non_zero_counts)
            
            # Find the qth percentile (default: 50th percentile)
            q = 0.5
            cumsum = np.cumsum(sorted_counts)
            total_sum = cumsum[-1]
            
            # Find the count value at which we've accumulated q of total
            threshold_idx = np.where(cumsum >= q * total_sum)[0][0]
            threshold_value = sorted_counts[threshold_idx]
            
            # Calculate scaling factor based on counts up to threshold
            counts_below_threshold = sample_counts[sample_counts <= threshold_value]
            scaling_factors[i] = np.sum(counts_below_threshold)
            
        # Normalize to median scaling factor
        scaling_factors = scaling_factors / np.median(scaling_factors[scaling_factors > 0])
        
        # Apply normalization
        for i in range(n_samples):
            if scaling_factors[i] > 0:
                normalized[i, :] = counts[i, :] / scaling_factors[i]
            else:
                normalized[i, :] = counts[i, :]
                
        self.normalization_factors = scaling_factors
        
        logger.info(f"CSS normalization complete. Scaling factors range: "
                   f"{np.min(scaling_factors):.3f} - {np.max(scaling_factors):.3f}")
        
        return normalized
    
    def _tmm_normalization(self, counts: np.ndarray, 
                           reference_sample: Optional[int] = None) -> np.ndarray:
        """Trimmed Mean of M-values normalization (edgeR-inspired).
        
        This method calculates normalization factors based on a trimmed mean
        of log-fold changes between samples.
        
        Args:
            counts: Raw count matrix (samples x features)
            reference_sample: Index of reference sample (None for auto-select)
            
        Returns:
            TMM-normalized counts
        """
        n_samples, n_features = counts.shape
        
        # Add pseudocount to avoid log(0)
        counts_pseudo = counts + 0.5
        
        # Select reference sample (sample with count distribution closest to mean)
        if reference_sample is None:
            mean_counts = np.mean(counts_pseudo, axis=0)
            distances = np.array([
                np.sum(np.abs(np.log2(counts_pseudo[i, :] / mean_counts))) 
                for i in range(n_samples)
            ])
            reference_sample = np.argmin(distances)
            
        logger.info(f"Using sample {reference_sample} as reference for TMM")
        
        # Calculate normalization factors
        normalization_factors = np.ones(n_samples)
        ref_counts = counts_pseudo[reference_sample, :]
        
        for i in range(n_samples):
            if i == reference_sample:
                continue
                
            sample_counts = counts_pseudo[i, :]
            
            # Calculate M-values (log-fold changes) and A-values (average expression)
            # Only for features present in both samples
            mask = (ref_counts > 0) & (sample_counts > 0)
            
            if np.sum(mask) < 10:
                logger.warning(f"Sample {i} has few features in common with reference")
                continue
                
            M_values = np.log2(sample_counts[mask] / ref_counts[mask])
            A_values = 0.5 * (np.log2(sample_counts[mask]) + np.log2(ref_counts[mask]))
            
            # Trim extreme values (default: 30% of M-values, 5% of A-values)
            M_trim = 0.3
            A_trim = 0.05
            
            # Remove extreme M-values
            M_lower = np.percentile(M_values, M_trim * 100 / 2)
            M_upper = np.percentile(M_values, 100 - M_trim * 100 / 2)
            
            # Remove extreme A-values
            A_lower = np.percentile(A_values, A_trim * 100)
            A_upper = np.percentile(A_values, 100 - A_trim * 100)
            
            # Keep only non-trimmed values
            keep_mask = (M_values >= M_lower) & (M_values <= M_upper) & \
                       (A_values >= A_lower) & (A_values <= A_upper)
                       
            if np.sum(keep_mask) < 10:
                logger.warning(f"Too few features remain after trimming for sample {i}")
                continue
                
            # Calculate weighted mean of M-values
            M_trimmed = M_values[keep_mask]
            weights = 1.0 / (1.0 / sample_counts[mask][keep_mask] + 
                           1.0 / ref_counts[mask][keep_mask])
            
            weighted_mean_M = np.sum(weights * M_trimmed) / np.sum(weights)
            normalization_factors[i] = 2 ** weighted_mean_M
            
        # Normalize factors to have geometric mean of 1
        log_factors = np.log(normalization_factors)
        log_factors = log_factors - np.mean(log_factors)
        normalization_factors = np.exp(log_factors)
        
        # Apply normalization
        normalized = counts / normalization_factors[:, np.newaxis]
        
        self.normalization_factors = normalization_factors
        
        logger.info(f"TMM normalization complete. Factor range: "
                   f"{np.min(normalization_factors):.3f} - {np.max(normalization_factors):.3f}")
        
        return normalized
    
    def _rle_normalization(self, counts: np.ndarray,
                          reference_sample: Optional[int] = None) -> np.ndarray:
        """Relative Log Expression normalization (DESeq2-inspired).
        
        This method assumes most features are not differentially abundant
        and calculates size factors based on the median of ratios.
        
        Args:
            counts: Raw count matrix (samples x features)
            reference_sample: Index of reference sample (None for geometric mean)
            
        Returns:
            RLE-normalized counts
        """
        n_samples, n_features = counts.shape
        
        # Calculate reference (geometric mean across samples for each feature)
        # Add pseudocount to handle zeros
        counts_pseudo = counts + 0.5
        
        if reference_sample is None:
            # Use geometric mean as reference
            log_counts = np.log(counts_pseudo)
            geometric_mean = np.exp(np.mean(log_counts, axis=0))
        else:
            # Use specified sample as reference
            geometric_mean = counts_pseudo[reference_sample, :]
            
        # Calculate size factors
        size_factors = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Calculate ratios to reference
            ratios = counts_pseudo[i, :] / geometric_mean
            
            # Remove zeros and infinities
            valid_ratios = ratios[(ratios > 0) & np.isfinite(ratios)]
            
            if len(valid_ratios) > 0:
                # Size factor is median of ratios
                size_factors[i] = np.median(valid_ratios)
            else:
                size_factors[i] = 1.0
                logger.warning(f"No valid ratios for sample {i}, using size factor 1.0")
                
        # Normalize size factors to have geometric mean of 1
        log_factors = np.log(size_factors)
        log_factors = log_factors - np.mean(log_factors)
        size_factors = np.exp(log_factors)
        
        # Apply normalization
        normalized = counts / size_factors[:, np.newaxis]
        
        self.normalization_factors = size_factors
        
        logger.info(f"RLE normalization complete. Size factor range: "
                   f"{np.min(size_factors):.3f} - {np.max(size_factors):.3f}")
        
        return normalized
    
    def _tss_normalization(self, counts: np.ndarray) -> np.ndarray:
        """Total Sum Scaling normalization (relative abundance).
        
        Simple normalization by total counts per sample.
        
        Args:
            counts: Raw count matrix (samples x features)
            
        Returns:
            TSS-normalized counts (relative abundances)
        """
        # Calculate total counts per sample
        total_counts = np.sum(counts, axis=1)
        
        # Avoid division by zero
        total_counts[total_counts == 0] = 1.0
        
        # Normalize to relative abundance
        normalized = counts / total_counts[:, np.newaxis]
        
        self.normalization_factors = total_counts
        
        logger.info("TSS normalization complete (relative abundance)")
        
        return normalized
    
    def _clr_normalization(self, counts: np.ndarray) -> np.ndarray:
        """Centered Log-Ratio transformation.
        
        This method transforms compositional data to remove the
        constraint that abundances sum to a constant.
        
        Args:
            counts: Raw count matrix (samples x features)
            
        Returns:
            CLR-transformed counts
        """
        # Add pseudocount to handle zeros
        counts_pseudo = counts + 0.5
        
        # Calculate CLR transformation for each sample
        normalized = np.zeros_like(counts, dtype=float)
        
        for i in range(counts.shape[0]):
            sample_counts = counts_pseudo[i, :]
            
            # Log transform
            log_counts = np.log(sample_counts)
            
            # Center by geometric mean
            geometric_mean = np.mean(log_counts)
            normalized[i, :] = log_counts - geometric_mean
            
        logger.info("CLR transformation complete")
        
        return normalized
    
    def evaluate_normalization(self, raw_counts: np.ndarray, 
                             normalized_counts: np.ndarray,
                             metadata: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Evaluate the effectiveness of normalization.
        
        Args:
            raw_counts: Original count matrix
            normalized_counts: Normalized count matrix
            metadata: Optional metadata for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Calculate coefficient of variation reduction
        cv_raw = np.std(raw_counts, axis=0) / (np.mean(raw_counts, axis=0) + 1e-10)
        cv_norm = np.std(normalized_counts, axis=0) / (np.mean(normalized_counts, axis=0) + 1e-10)
        
        metrics['cv_reduction'] = np.median(cv_raw - cv_norm)
        metrics['cv_reduction_pct'] = np.median((cv_raw - cv_norm) / cv_raw) * 100
        
        # Calculate dispersion statistics
        metrics['raw_dispersion'] = np.mean(np.var(raw_counts, axis=0))
        metrics['normalized_dispersion'] = np.mean(np.var(normalized_counts, axis=0))
        
        # If we have normalization factors, check their distribution
        if self.normalization_factors is not None:
            metrics['factor_cv'] = np.std(self.normalization_factors) / np.mean(self.normalization_factors)
            metrics['factor_range'] = np.max(self.normalization_factors) / np.min(self.normalization_factors)
            
        logger.info(f"Normalization evaluation: CV reduction = {metrics['cv_reduction_pct']:.1f}%")
        
        return metrics
    
    def normalize_for_depth(self, count_matrix: Union[np.ndarray, pd.DataFrame], 
                           sample_depths: Optional[pd.Series] = None, 
                           method: str = "subsampling") -> Union[np.ndarray, pd.DataFrame]:
        """Normalize count data to account for varying sequencing depths.
        
        Args:
            count_matrix: Feature counts for each sample (samples x features)
            sample_depths: Total reads per sample (calculated if None)
            method: Normalization method ('subsampling', 'scaling', 'rarefaction')
            
        Returns:
            Depth-normalized count matrix
        """
        # Convert to appropriate format
        if isinstance(count_matrix, pd.DataFrame):
            counts_array = count_matrix.values
            return_df = True
            df_index = count_matrix.index
            df_columns = count_matrix.columns
        else:
            counts_array = count_matrix.copy()
            return_df = False
            
        # Calculate sample depths if not provided
        if sample_depths is None:
            sample_depths = np.sum(counts_array, axis=1)
        elif isinstance(sample_depths, pd.Series):
            sample_depths = sample_depths.values
            
        # Check for depth heterogeneity
        depth_cv = np.std(sample_depths) / np.mean(sample_depths)
        if depth_cv > 0.5:
            logger.warning(f"High sequencing depth heterogeneity detected (CV={depth_cv:.2f})")
            
        # Apply depth normalization
        if method == "subsampling":
            normalized = self._subsample_normalization(counts_array, sample_depths)
        elif method == "scaling":
            normalized = self._scaling_normalization(counts_array, sample_depths)
        elif method == "rarefaction":
            normalized = self._rarefaction_normalization(counts_array, sample_depths)
        else:
            raise ValueError(f"Unknown depth normalization method: {method}")
            
        # Test for depth-similarity correlation
        self._test_depth_similarity_correlation(normalized, sample_depths)
            
        # Return in original format
        if return_df:
            return pd.DataFrame(normalized, index=df_index, columns=df_columns)
        return normalized
    
    def _subsample_normalization(self, counts: np.ndarray, sample_depths: np.ndarray) -> np.ndarray:
        """Subsample reads to minimum depth across samples."""
        min_depth = int(np.min(sample_depths[sample_depths > 0]))
        logger.info(f"Subsampling all samples to {min_depth} reads")
        
        normalized = np.zeros_like(counts, dtype=float)
        
        for i, depth in enumerate(sample_depths):
            if depth == 0:
                continue
                
            # Calculate subsampling probability
            prob = min_depth / depth
            
            if prob >= 1.0:
                # No subsampling needed
                normalized[i, :] = counts[i, :]
            else:
                # Subsample using binomial distribution
                normalized[i, :] = np.random.binomial(counts[i, :].astype(int), prob)
                
        return normalized
    
    def _scaling_normalization(self, counts: np.ndarray, sample_depths: np.ndarray) -> np.ndarray:
        """Scale counts to median depth across samples."""
        median_depth = np.median(sample_depths[sample_depths > 0])
        logger.info(f"Scaling all samples to median depth {median_depth:.0f}")
        
        scaling_factors = median_depth / sample_depths
        scaling_factors[sample_depths == 0] = 1.0  # Avoid division by zero
        
        normalized = counts * scaling_factors[:, np.newaxis]
        return normalized
    
    def _rarefaction_normalization(self, counts: np.ndarray, sample_depths: np.ndarray) -> np.ndarray:
        """Rarefy samples to minimum depth with multiple iterations."""
        min_depth = int(np.min(sample_depths[sample_depths > 0]))
        n_iterations = 10  # Number of rarefaction iterations
        
        logger.info(f"Rarefying samples to {min_depth} reads with {n_iterations} iterations")
        
        normalized_sum = np.zeros_like(counts, dtype=float)
        
        for iteration in range(n_iterations):
            iteration_counts = np.zeros_like(counts, dtype=float)
            
            for i, depth in enumerate(sample_depths):
                if depth == 0:
                    continue
                    
                if depth <= min_depth:
                    iteration_counts[i, :] = counts[i, :]
                else:
                    # Randomly sample min_depth reads
                    prob_vector = counts[i, :] / depth
                    iteration_counts[i, :] = np.random.multinomial(min_depth, prob_vector)
                    
            normalized_sum += iteration_counts
            
        # Average across iterations
        normalized = normalized_sum / n_iterations
        return normalized
    
    def _test_depth_similarity_correlation(self, normalized_counts: np.ndarray, 
                                         sample_depths: np.ndarray) -> None:
        """Test if sample similarity correlates with sequencing depth."""
        from scipy.stats import pearsonr
        from scipy.spatial.distance import pdist, squareform
        
        # Calculate pairwise similarities (using Bray-Curtis)
        distances = pdist(normalized_counts, metric='braycurtis')
        similarities = 1 - distances
        
        # Calculate pairwise depth differences
        depth_matrix = np.abs(sample_depths[:, np.newaxis] - sample_depths[np.newaxis, :])
        depth_diffs = squareform(depth_matrix, checks=False)
        
        # Test correlation
        correlation, p_value = pearsonr(similarities, -depth_diffs)  # Negative because larger diff = less similarity
        
        if p_value < 0.05:
            logger.warning(f"Sample similarity correlates with sequencing depth "
                          f"(r={correlation:.3f}, p={p_value:.3f})")
        else:
            logger.info(f"No significant depth-similarity correlation detected "
                       f"(r={correlation:.3f}, p={p_value:.3f})")
    
    def detect_depth_heterogeneity(self, count_matrix: Union[np.ndarray, pd.DataFrame]) -> Dict[str, float]:
        """Detect and report varying sequencing depths across samples.
        
        Args:
            count_matrix: Feature counts for each sample
            
        Returns:
            Dictionary with depth heterogeneity metrics
        """
        if isinstance(count_matrix, pd.DataFrame):
            counts_array = count_matrix.values
        else:
            counts_array = count_matrix
            
        sample_depths = np.sum(counts_array, axis=1)
        
        metrics = {
            'min_depth': float(np.min(sample_depths)),
            'max_depth': float(np.max(sample_depths)),
            'median_depth': float(np.median(sample_depths)),
            'mean_depth': float(np.mean(sample_depths)),
            'depth_cv': float(np.std(sample_depths) / np.mean(sample_depths)),
            'depth_range_ratio': float(np.max(sample_depths) / np.min(sample_depths[sample_depths > 0])),
            'samples_below_1000': int(np.sum(sample_depths < 1000)),
            'samples_above_100k': int(np.sum(sample_depths > 100000))
        }
        
        # Log warnings for problematic depths
        if metrics['depth_cv'] > 0.5:
            logger.warning(f"High depth variability detected (CV={metrics['depth_cv']:.2f})")
        if metrics['depth_range_ratio'] > 10:
            logger.warning(f"Large depth range detected (ratio={metrics['depth_range_ratio']:.1f})")
        if metrics['samples_below_1000'] > 0:
            logger.warning(f"{metrics['samples_below_1000']} samples have <1000 reads")
            
        return metrics