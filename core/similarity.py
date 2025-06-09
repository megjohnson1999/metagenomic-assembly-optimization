"""Bias-aware distance metrics for metagenomic sample comparison."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import pairwise_distances
import logging

logger = logging.getLogger(__name__)


class BiasAwareSimilarity:
    """Calculate bias-aware distance metrics for metagenomic samples.
    
    This class implements distance metrics that are robust to technical
    variation while preserving biological signal.
    """
    
    def __init__(self):
        """Initialize the bias-aware similarity calculator."""
        self.distance_matrix = None
        self.metric_used = None
        
    def calculate_distances(self, data: Union[np.ndarray, pd.DataFrame],
                          metric: str = 'jensen_shannon',
                          presence_absence: bool = False,
                          weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
        """Calculate pairwise distances between samples.
        
        Args:
            data: Count matrix (samples x features)
            metric: Distance metric to use
            presence_absence: Convert to presence/absence before distance calculation
            weights: Optional feature weights for weighted distances
            
        Returns:
            Tuple of (distance matrix, sample names)
        """
        # Convert to numpy array if DataFrame
        if isinstance(data, pd.DataFrame):
            data_array = data.values
            sample_names = list(data.index)
        else:
            data_array = data.copy()
            sample_names = [f"Sample_{i}" for i in range(data.shape[0])]
            
        # Apply presence/absence transformation if requested
        if presence_absence:
            data_array = (data_array > 0).astype(float)
            logger.info("Applied presence/absence transformation")
            
        # Calculate distances
        if metric == 'jensen_shannon':
            dist_matrix = self._jensen_shannon_distance(data_array)
        elif metric == 'robust_bray_curtis':
            dist_matrix = self._robust_bray_curtis_distance(data_array)
        elif metric == 'weighted_unifrac':
            dist_matrix = self._weighted_unifrac_distance(data_array, weights)
        elif metric == 'aitchison':
            dist_matrix = self._aitchison_distance(data_array)
        elif metric == 'hellinger':
            dist_matrix = self._hellinger_distance(data_array)
        elif metric == 'chi_squared':
            dist_matrix = self._chi_squared_distance(data_array)
        else:
            raise ValueError(f"Unknown distance metric: {metric}")
            
        self.distance_matrix = dist_matrix
        self.metric_used = metric
        
        logger.info(f"Calculated {metric} distances for {len(sample_names)} samples")
        
        return dist_matrix, sample_names
    
    def _jensen_shannon_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate Jensen-Shannon distance matrix.
        
        JS distance is the square root of JS divergence and is a proper
        metric that is robust to sampling effects.
        
        Args:
            data: Count matrix (samples x features)
            
        Returns:
            Distance matrix
        """
        n_samples = data.shape[0]
        dist_matrix = np.zeros((n_samples, n_samples))
        
        # Convert to probability distributions
        prob_data = data / (np.sum(data, axis=1, keepdims=True) + 1e-10)
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                p = prob_data[i, :]
                q = prob_data[j, :]
                
                # Calculate JS divergence
                m = 0.5 * (p + q)
                
                # Handle zeros with small pseudocount
                p_safe = np.where(p > 0, p, 1e-10)
                q_safe = np.where(q > 0, q, 1e-10)
                m_safe = np.where(m > 0, m, 1e-10)
                
                # KL divergences
                kl_pm = np.sum(p * np.log(p_safe / m_safe))
                kl_qm = np.sum(q * np.log(q_safe / m_safe))
                
                # JS divergence
                js_div = 0.5 * kl_pm + 0.5 * kl_qm
                
                # JS distance is square root of divergence
                js_dist = np.sqrt(np.maximum(0, js_div))
                
                dist_matrix[i, j] = js_dist
                dist_matrix[j, i] = js_dist
                
        return dist_matrix
    
    def _robust_bray_curtis_distance(self, data: np.ndarray,
                                   trim_fraction: float = 0.1) -> np.ndarray:
        """Calculate robust Bray-Curtis distance with outlier trimming.
        
        This version trims extreme values before calculating BC distance
        to reduce the impact of technical outliers.
        
        Args:
            data: Count matrix (samples x features)
            trim_fraction: Fraction of extreme values to trim per sample
            
        Returns:
            Distance matrix
        """
        n_samples, n_features = data.shape
        dist_matrix = np.zeros((n_samples, n_samples))
        
        # Trim extreme values in each sample
        trimmed_data = np.zeros_like(data)
        
        for i in range(n_samples):
            sample = data[i, :]
            
            # Find trimming thresholds
            if trim_fraction > 0:
                lower_thresh = np.percentile(sample[sample > 0], 
                                           trim_fraction * 100 / 2)
                upper_thresh = np.percentile(sample, 
                                           100 - trim_fraction * 100 / 2)
                
                # Apply trimming
                trimmed_sample = np.clip(sample, lower_thresh, upper_thresh)
            else:
                trimmed_sample = sample
                
            trimmed_data[i, :] = trimmed_sample
        
        # Calculate Bray-Curtis on trimmed data
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                u = trimmed_data[i, :]
                v = trimmed_data[j, :]
                
                numerator = np.sum(np.abs(u - v))
                denominator = np.sum(u) + np.sum(v)
                
                if denominator > 0:
                    bc_dist = numerator / denominator
                else:
                    bc_dist = 0.0
                    
                dist_matrix[i, j] = bc_dist
                dist_matrix[j, i] = bc_dist
                
        return dist_matrix
    
    def _weighted_unifrac_distance(self, data: np.ndarray,
                                  weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate weighted UniFrac-inspired distance for k-mer data.
        
        This adapts the UniFrac concept to k-mer data by using feature
        abundances and optional phylogenetic-like weights.
        
        Args:
            data: Count matrix (samples x features)
            weights: Optional feature weights (e.g., based on k-mer complexity)
            
        Returns:
            Distance matrix
        """
        n_samples, n_features = data.shape
        
        # If no weights provided, use equal weights
        if weights is None:
            weights = np.ones(n_features)
        elif len(weights) != n_features:
            raise ValueError("Weights length must match number of features")
            
        # Normalize to relative abundances
        rel_data = data / (np.sum(data, axis=1, keepdims=True) + 1e-10)
        
        dist_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                p_i = rel_data[i, :]
                p_j = rel_data[j, :]
                
                # Weighted numerator (absolute differences)
                numerator = np.sum(weights * np.abs(p_i - p_j))
                
                # Weighted denominator (total abundance)
                denominator = np.sum(weights * (p_i + p_j))
                
                if denominator > 0:
                    unifrac_dist = numerator / denominator
                else:
                    unifrac_dist = 0.0
                    
                dist_matrix[i, j] = unifrac_dist
                dist_matrix[j, i] = unifrac_dist
                
        return dist_matrix
    
    def _aitchison_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate Aitchison distance for compositional data.
        
        This is the Euclidean distance in log-ratio space, appropriate
        for compositional data like relative abundances.
        
        Args:
            data: Count matrix (samples x features)
            
        Returns:
            Distance matrix
        """
        # Add pseudocount to handle zeros
        data_pseudo = data + 0.5
        
        # Convert to relative abundances
        rel_data = data_pseudo / np.sum(data_pseudo, axis=1, keepdims=True)
        
        # CLR transformation
        clr_data = np.log(rel_data) - np.mean(np.log(rel_data), axis=1, keepdims=True)
        
        # Calculate Euclidean distance in CLR space
        dist_matrix = pairwise_distances(clr_data, metric='euclidean')
        
        return dist_matrix
    
    def _hellinger_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate Hellinger distance.
        
        Hellinger distance is robust to sampling variation and is
        a proper metric for probability distributions.
        
        Args:
            data: Count matrix (samples x features)
            
        Returns:
            Distance matrix
        """
        # Convert to relative abundances
        rel_data = data / (np.sum(data, axis=1, keepdims=True) + 1e-10)
        
        # Square root transformation
        sqrt_data = np.sqrt(rel_data)
        
        n_samples = data.shape[0]
        dist_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # Hellinger distance
                hellinger_dist = np.sqrt(0.5 * np.sum((sqrt_data[i, :] - sqrt_data[j, :]) ** 2))
                
                dist_matrix[i, j] = hellinger_dist
                dist_matrix[j, i] = hellinger_dist
                
        return dist_matrix
    
    def _chi_squared_distance(self, data: np.ndarray) -> np.ndarray:
        """Calculate Chi-squared distance.
        
        Chi-squared distance is useful for count data and can be
        less sensitive to rare features than other metrics.
        
        Args:
            data: Count matrix (samples x features)
            
        Returns:
            Distance matrix
        """
        n_samples = data.shape[0]
        dist_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                u = data[i, :]
                v = data[j, :]
                
                # Chi-squared distance
                # Sum over features of (u_i - v_i)^2 / (u_i + v_i)
                sum_uv = u + v
                mask = sum_uv > 0
                
                if np.any(mask):
                    chi_sq_dist = np.sqrt(np.sum(((u[mask] - v[mask]) ** 2) / sum_uv[mask]))
                else:
                    chi_sq_dist = 0.0
                    
                dist_matrix[i, j] = chi_sq_dist
                dist_matrix[j, i] = chi_sq_dist
                
        return dist_matrix
    
    def calculate_feature_weights(self, data: np.ndarray,
                                method: str = 'variance_stabilizing') -> np.ndarray:
        """Calculate feature weights for weighted distance metrics.
        
        Args:
            data: Count matrix (samples x features)
            method: Method for calculating weights
            
        Returns:
            Feature weights
        """
        if method == 'variance_stabilizing':
            # Weight inversely proportional to variance
            variances = np.var(data, axis=0)
            weights = 1.0 / (variances + 1e-10)
            
        elif method == 'prevalence':
            # Weight by feature prevalence (proportion of non-zero samples)
            prevalence = np.mean(data > 0, axis=0)
            weights = prevalence
            
        elif method == 'mean_abundance':
            # Weight by mean abundance
            weights = np.mean(data, axis=0)
            
        elif method == 'information_content':
            # Weight by information content (inverse of entropy)
            rel_data = data / (np.sum(data, axis=1, keepdims=True) + 1e-10)
            mean_rel = np.mean(rel_data, axis=0)
            entropy = -np.sum(mean_rel * np.log(mean_rel + 1e-10))
            weights = 1.0 / (entropy + 1e-10)
            
        else:
            raise ValueError(f"Unknown weighting method: {method}")
            
        # Normalize weights
        weights = weights / np.sum(weights)
        
        logger.info(f"Calculated {method} feature weights")
        
        return weights
    
    def evaluate_distance_robustness(self, data: np.ndarray,
                                   n_bootstrap: int = 100,
                                   sample_fraction: float = 0.8) -> Dict[str, float]:
        """Evaluate the robustness of distance calculations through bootstrap.
        
        Args:
            data: Count matrix (samples x features)
            n_bootstrap: Number of bootstrap iterations
            sample_fraction: Fraction of features to sample in each bootstrap
            
        Returns:
            Dictionary of robustness metrics
        """
        n_samples, n_features = data.shape
        n_subsample = int(sample_fraction * n_features)
        
        # Calculate original distances
        original_distances, _ = self.calculate_distances(data, self.metric_used)
        
        # Bootstrap sampling
        bootstrap_correlations = []
        
        for i in range(n_bootstrap):
            # Randomly sample features
            feature_indices = np.random.choice(n_features, n_subsample, replace=False)
            subsample_data = data[:, feature_indices]
            
            # Calculate distances on subsample
            bootstrap_distances, _ = self.calculate_distances(subsample_data, self.metric_used)
            
            # Calculate correlation with original
            orig_condensed = squareform(original_distances)
            boot_condensed = squareform(bootstrap_distances)
            
            correlation = stats.pearsonr(orig_condensed, boot_condensed)[0]
            bootstrap_correlations.append(correlation)
            
        bootstrap_correlations = np.array(bootstrap_correlations)
        
        metrics = {
            'mean_correlation': np.mean(bootstrap_correlations),
            'std_correlation': np.std(bootstrap_correlations),
            'min_correlation': np.min(bootstrap_correlations),
            'robustness_score': np.mean(bootstrap_correlations > 0.8)  # Fraction with high correlation
        }
        
        logger.info(f"Distance robustness: mean correlation = {metrics['mean_correlation']:.3f}")
        
        return metrics