"""Compositional data handling for metagenomic data analysis.

This module provides tools for handling the compositional nature of metagenomic data,
including transformations and appropriate distance metrics for compositional data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union, List
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import logging

logger = logging.getLogger(__name__)


class CompositionalDataHandler:
    """Handle compositional data transformations and analysis.
    
    This class implements methods specifically designed for compositional data
    where values represent proportions or relative abundances that sum to a constant.
    """
    
    def __init__(self):
        """Initialize the compositional data handler."""
        self.transformation_used = None
        self.pseudo_count = None
        
    def transform_compositional_data(self, count_matrix: Union[np.ndarray, pd.DataFrame], 
                                   transformation: str = "clr", 
                                   pseudo_count: float = 0.5) -> Union[np.ndarray, pd.DataFrame]:
        """Apply compositional data transformations to count data.
        
        Args:
            count_matrix: Feature counts for each sample (samples x features)
            transformation: Transformation type ('clr', 'alr', 'ilr', 'proportion')
            pseudo_count: Value to add to counts to handle zeros
            
        Returns:
            Transformed count matrix
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
            
        # Store parameters
        self.transformation_used = transformation
        self.pseudo_count = pseudo_count
        
        # Validate compositional data
        self._validate_compositional_data(counts_array)
        
        # Apply transformation
        if transformation == 'clr':
            transformed = self._clr_transformation(counts_array, pseudo_count)
        elif transformation == 'alr':
            transformed = self._alr_transformation(counts_array, pseudo_count)
        elif transformation == 'ilr':
            transformed = self._ilr_transformation(counts_array, pseudo_count)
        elif transformation == 'proportion':
            transformed = self._proportion_transformation(counts_array)
        else:
            raise ValueError(f"Unknown transformation: {transformation}")
            
        # Return in original format
        if return_df:
            if transformation == 'alr':
                # ALR reduces dimensionality by 1
                new_columns = df_columns[:-1]  # Remove last column (reference)
                return pd.DataFrame(transformed, index=df_index, columns=new_columns)
            elif transformation == 'ilr':
                # ILR creates new coordinate system
                new_columns = [f"ILR_{i+1}" for i in range(transformed.shape[1])]
                return pd.DataFrame(transformed, index=df_index, columns=new_columns)
            else:
                return pd.DataFrame(transformed, index=df_index, columns=df_columns)
        return transformed
    
    def _clr_transformation(self, counts: np.ndarray, pseudo_count: float) -> np.ndarray:
        """Centered Log-Ratio transformation.
        
        CLR transforms compositional data to remove the sum constraint while
        preserving all the information in the original composition.
        
        Args:
            counts: Raw count matrix (samples x features)
            pseudo_count: Value to add to handle zeros
            
        Returns:
            CLR-transformed data
        """
        # Add pseudocount and convert to proportions
        counts_pseudo = counts + pseudo_count
        proportions = counts_pseudo / np.sum(counts_pseudo, axis=1)[:, np.newaxis]
        
        # Calculate CLR transformation
        log_proportions = np.log(proportions)
        geometric_mean = np.mean(log_proportions, axis=1)
        clr_transformed = log_proportions - geometric_mean[:, np.newaxis]
        
        logger.info(f"CLR transformation applied with pseudo_count={pseudo_count}")
        
        return clr_transformed
    
    def _alr_transformation(self, counts: np.ndarray, pseudo_count: float) -> np.ndarray:
        """Additive Log-Ratio transformation.
        
        ALR uses the last component as a reference and expresses all other
        components as log-ratios relative to this reference.
        
        Args:
            counts: Raw count matrix (samples x features)
            pseudo_count: Value to add to handle zeros
            
        Returns:
            ALR-transformed data (dimensionality reduced by 1)
        """
        # Add pseudocount
        counts_pseudo = counts + pseudo_count
        
        # Use last column as reference
        reference = counts_pseudo[:, -1]
        
        # Calculate log-ratios for all other columns
        alr_transformed = np.log(counts_pseudo[:, :-1] / reference[:, np.newaxis])
        
        logger.info(f"ALR transformation applied with last feature as reference")
        
        return alr_transformed
    
    def _ilr_transformation(self, counts: np.ndarray, pseudo_count: float) -> np.ndarray:
        """Isometric Log-Ratio transformation.
        
        ILR creates an orthonormal coordinate system in the simplex,
        preserving distances and avoiding the arbitrariness of ALR.
        
        Args:
            counts: Raw count matrix (samples x features)
            pseudo_count: Value to add to handle zeros
            
        Returns:
            ILR-transformed data (dimensionality reduced by 1)
        """
        n_features = counts.shape[1]
        
        # Add pseudocount and convert to proportions
        counts_pseudo = counts + pseudo_count
        proportions = counts_pseudo / np.sum(counts_pseudo, axis=1)[:, np.newaxis]
        
        # Create Helmert matrix for ILR transformation
        helmert_matrix = self._create_helmert_matrix(n_features)
        
        # Apply ILR transformation
        log_proportions = np.log(proportions)
        ilr_transformed = log_proportions @ helmert_matrix
        
        logger.info(f"ILR transformation applied")
        
        return ilr_transformed
    
    def _proportion_transformation(self, counts: np.ndarray) -> np.ndarray:
        """Simple proportion transformation (relative abundance).
        
        Args:
            counts: Raw count matrix (samples x features)
            
        Returns:
            Proportion-transformed data
        """
        total_counts = np.sum(counts, axis=1)
        total_counts[total_counts == 0] = 1.0  # Avoid division by zero
        
        proportions = counts / total_counts[:, np.newaxis]
        
        logger.info("Proportion transformation applied")
        
        return proportions
    
    def _create_helmert_matrix(self, n_features: int) -> np.ndarray:
        """Create Helmert matrix for ILR transformation.
        
        Args:
            n_features: Number of features
            
        Returns:
            Helmert matrix of shape (n_features, n_features-1)
        """
        helmert = np.zeros((n_features, n_features - 1))
        
        for i in range(n_features - 1):
            # Standard Helmert matrix construction
            # Set first i+1 elements to sqrt(1/(i+1)/(i+2))
            helmert[:i+1, i] = 1.0 / np.sqrt((i + 1) * (i + 2))
            # Set the (i+1)th element to -sqrt((i+1)/(i+2))
            if i + 1 < n_features:
                helmert[i + 1, i] = -np.sqrt((i + 1) / (i + 2))
            
        return helmert
    
    def calculate_aitchison_distance(self, data: Union[np.ndarray, pd.DataFrame],
                                   pseudo_count: float = 0.5) -> Tuple[np.ndarray, List[str]]:
        """Calculate Aitchison distance matrix for compositional data.
        
        The Aitchison distance is the Euclidean distance in the CLR-transformed space,
        which is the natural distance metric for compositional data.
        
        Args:
            data: Count or proportion data (samples x features)
            pseudo_count: Value to add to handle zeros
            
        Returns:
            Tuple of (distance_matrix, sample_names)
        """
        if isinstance(data, pd.DataFrame):
            sample_names = list(data.index)
            data_array = data.values
        else:
            sample_names = [f"Sample_{i}" for i in range(data.shape[0])]
            data_array = data
            
        # Transform to CLR space
        clr_data = self._clr_transformation(data_array, pseudo_count)
        
        # Calculate Euclidean distances in CLR space
        distances = pdist(clr_data, metric='euclidean')
        distance_matrix = squareform(distances)
        
        logger.info("Aitchison distance matrix calculated")
        
        return distance_matrix, sample_names
    
    def detect_compositional_issues(self, count_matrix: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Union[float, int]]:
        """Detect potential issues with compositional data.
        
        Args:
            count_matrix: Feature counts for each sample
            
        Returns:
            Dictionary with compositional data quality metrics
        """
        if isinstance(count_matrix, pd.DataFrame):
            counts_array = count_matrix.values
        else:
            counts_array = count_matrix
            
        n_samples, n_features = counts_array.shape
        
        # Calculate basic statistics
        zero_fraction = np.mean(counts_array == 0)
        sparse_samples = np.sum(np.sum(counts_array > 0, axis=1) < n_features * 0.1)
        
        # Check for samples with very few non-zero features
        features_per_sample = np.sum(counts_array > 0, axis=1)
        min_features = np.min(features_per_sample)
        
        # Check for highly variable total counts (depth heterogeneity)
        total_counts = np.sum(counts_array, axis=1)
        depth_cv = np.std(total_counts) / np.mean(total_counts) if np.mean(total_counts) > 0 else 0
        
        # Check for features with extreme prevalence
        feature_prevalence = np.sum(counts_array > 0, axis=0) / n_samples
        rare_features = np.sum(feature_prevalence < 0.05)
        ubiquitous_features = np.sum(feature_prevalence > 0.95)
        
        metrics = {
            'zero_fraction': float(zero_fraction),
            'sparse_samples': int(sparse_samples),
            'min_features_per_sample': int(min_features),
            'depth_cv': float(depth_cv),
            'rare_features': int(rare_features),
            'ubiquitous_features': int(ubiquitous_features),
            'n_samples': int(n_samples),
            'n_features': int(n_features)
        }
        
        # Log warnings for potential issues
        if zero_fraction > 0.7:
            logger.warning(f"High sparsity detected: {zero_fraction:.1%} of values are zero")
        if sparse_samples > n_samples * 0.1:
            logger.warning(f"{sparse_samples} samples have very few features")
        if depth_cv > 0.5:
            logger.warning(f"High depth variability may affect compositional analysis")
        if rare_features > n_features * 0.5:
            logger.warning(f"Many rare features ({rare_features}) may cause instability")
            
        return metrics
    
    def _validate_compositional_data(self, counts: np.ndarray) -> None:
        """Validate that data is appropriate for compositional analysis.
        
        Args:
            counts: Count matrix to validate
        """
        # Check for negative values
        if np.any(counts < 0):
            raise ValueError("Compositional data cannot contain negative values")
            
        # Check for all-zero samples
        zero_samples = np.sum(counts, axis=1) == 0
        if np.any(zero_samples):
            n_zero = np.sum(zero_samples)
            logger.warning(f"{n_zero} samples have zero total counts")
            
        # Check for all-zero features
        zero_features = np.sum(counts, axis=0) == 0
        if np.any(zero_features):
            n_zero = np.sum(zero_features)
            logger.warning(f"{n_zero} features have zero total counts")
    
    def evaluate_transformation_quality(self, original_data: np.ndarray, 
                                      transformed_data: np.ndarray) -> Dict[str, float]:
        """Evaluate the quality of a compositional transformation.
        
        Args:
            original_data: Original count data
            transformed_data: Transformed data
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        # Calculate variance explained by transformation
        original_var = np.var(original_data, axis=0)
        transformed_var = np.var(transformed_data, axis=0)
        
        mean_original_var = np.mean(original_var)
        if mean_original_var > 0:
            metrics['variance_ratio'] = np.mean(transformed_var) / mean_original_var
        else:
            metrics['variance_ratio'] = 1.0
        
        # Check for outliers in transformed space
        z_scores = np.abs(stats.zscore(transformed_data, axis=0))
        outlier_fraction = np.mean(z_scores > 3)
        metrics['outlier_fraction'] = float(outlier_fraction)
        
        # Calculate effective dimensionality
        eigenvals = np.linalg.eigvals(np.cov(transformed_data.T))
        # Take only real, positive eigenvalues
        eigenvals = np.real(eigenvals[np.real(eigenvals) > 0])
        if len(eigenvals) > 0:
            normalized_eigenvals = eigenvals / np.sum(eigenvals)
            effective_dim = np.exp(stats.entropy(normalized_eigenvals))
            metrics['effective_dimensionality'] = float(effective_dim)
        else:
            metrics['effective_dimensionality'] = 0.0
        
        return metrics
    
    def recommend_transformation(self, count_matrix: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Union[str, float]]:
        """Recommend appropriate transformation based on data characteristics.
        
        Args:
            count_matrix: Feature counts for each sample
            
        Returns:
            Dictionary with transformation recommendation and reasoning
        """
        issues = self.detect_compositional_issues(count_matrix)
        
        recommendation = {
            'recommended_transformation': 'clr',
            'recommended_pseudo_count': 0.5,
            'reasoning': []
        }
        
        # Adjust recommendation based on data characteristics
        if issues['zero_fraction'] > 0.8:
            recommendation['recommended_transformation'] = 'proportion'
            recommendation['reasoning'].append("High sparsity suggests simple proportions")
            
        if issues['depth_cv'] > 1.0:
            recommendation['recommended_pseudo_count'] = 1.0
            recommendation['reasoning'].append("High depth variability requires larger pseudo-count")
            
        if issues['n_features'] < 10:
            recommendation['recommended_transformation'] = 'alr'
            recommendation['reasoning'].append("Few features suggest ALR transformation")
            
        if issues['rare_features'] > issues['n_features'] * 0.7:
            recommendation['recommended_pseudo_count'] = 1.0
            recommendation['reasoning'].append("Many rare features require larger pseudo-count")
            
        if not recommendation['reasoning']:
            recommendation['reasoning'].append("CLR is generally recommended for compositional data")
            
        return recommendation