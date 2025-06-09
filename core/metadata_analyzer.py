"""Metadata correlation analysis for sample grouping."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class MetadataAnalyzer:
    """Analyze correlations between metadata variables and sequence similarity.
    
    This class provides methods to identify metadata variables that correlate
    with sequence-based sample distances, helping to identify meaningful
    sample groupings.
    """
    
    def __init__(self):
        """Initialize the metadata analyzer."""
        self.metadata = None
        self.distance_matrix = None
        self.sample_names = None
        
    def load_metadata(self, filepath: Union[str, Path], sample_id_column: str = 'sample_id') -> pd.DataFrame:
        """Load and validate metadata from CSV/TSV file.
        
        Args:
            filepath: Path to metadata file
            sample_id_column: Name of the column containing sample identifiers
            
        Returns:
            Loaded metadata DataFrame
        """
        filepath = Path(filepath)
        
        # Determine separator
        if filepath.suffix.lower() in ['.tsv', '.txt']:
            sep = '\t'
        else:
            sep = ','
        
        try:
            self.metadata = pd.read_csv(filepath, sep=sep)
            
            # Validate sample ID column exists
            if sample_id_column not in self.metadata.columns:
                raise ValueError(f"Sample ID column '{sample_id_column}' not found in metadata")
            
            # Set sample ID as index
            self.metadata.set_index(sample_id_column, inplace=True)
            
            logger.info(f"Loaded metadata for {len(self.metadata)} samples with {len(self.metadata.columns)} variables")
            
            return self.metadata
            
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise
    
    def set_distance_matrix(self, distance_matrix: np.ndarray, sample_names: List[str]):
        """Set the sequence-based distance matrix.
        
        Args:
            distance_matrix: Square distance matrix
            sample_names: List of sample names corresponding to matrix rows/columns
        """
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("Distance matrix must be square")
        
        if len(sample_names) != distance_matrix.shape[0]:
            raise ValueError("Number of sample names must match distance matrix dimensions")
        
        self.distance_matrix = distance_matrix
        self.sample_names = sample_names
    
    def _validate_samples(self) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
        """Validate and align samples between metadata and distance matrix."""
        if self.metadata is None:
            raise ValueError("Metadata not loaded")
        
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not set")
        
        # Find common samples
        common_samples = list(set(self.sample_names) & set(self.metadata.index))
        
        if len(common_samples) == 0:
            raise ValueError("No common samples found between metadata and distance matrix")
        
        logger.info(f"Found {len(common_samples)} common samples")
        
        # Align data
        sample_indices = [self.sample_names.index(s) for s in common_samples]
        aligned_distance = self.distance_matrix[np.ix_(sample_indices, sample_indices)]
        aligned_metadata = self.metadata.loc[common_samples]
        
        return common_samples, aligned_distance, aligned_metadata
    
    def analyze_correlations(self, method: str = 'mantel', 
                           permutations: int = 999) -> Dict[str, Dict[str, float]]:
        """Analyze correlations between metadata variables and sequence distances.
        
        Args:
            method: Correlation method ('mantel' or 'anosim')
            permutations: Number of permutations for significance testing
            
        Returns:
            Dictionary with correlation results for each metadata variable
        """
        samples, distances, metadata = self._validate_samples()
        results = {}
        
        for column in metadata.columns:
            # Skip columns with all NaN values
            if metadata[column].isna().all():
                logger.warning(f"Skipping column '{column}' - all values are NaN")
                continue
            
            # Determine variable type
            if metadata[column].dtype in ['float64', 'int64']:
                # Continuous variable
                result = self._analyze_continuous_variable(
                    metadata[column], distances, method, permutations
                )
            else:
                # Categorical variable
                result = self._analyze_categorical_variable(
                    metadata[column], distances, method, permutations
                )
            
            if result:
                results[column] = result
        
        return results
    
    def _analyze_continuous_variable(self, variable: pd.Series, distances: np.ndarray,
                                   method: str, permutations: int) -> Dict[str, float]:
        """Analyze correlation for continuous variables."""
        # Remove NaN values
        valid_mask = ~variable.isna()
        if valid_mask.sum() < 3:
            logger.warning(f"Insufficient valid values for variable '{variable.name}'")
            return None
        
        valid_indices = np.where(valid_mask)[0]
        valid_values = variable[valid_mask].values
        valid_distances = distances[np.ix_(valid_indices, valid_indices)]
        
        if method == 'mantel':
            # Create distance matrix from continuous variable
            value_distances = squareform(pdist(valid_values.reshape(-1, 1)))
            
            # Mantel test
            r, p_value = self._mantel_test(valid_distances, value_distances, permutations)
            
            return {
                'type': 'continuous',
                'method': 'mantel',
                'statistic': r,
                'p_value': p_value,
                'n_samples': len(valid_values)
            }
        else:
            # For continuous variables with ANOSIM, bin into categories
            n_bins = min(4, len(np.unique(valid_values)))
            binned = pd.qcut(valid_values, n_bins, labels=False, duplicates='drop')
            
            if len(np.unique(binned)) < 2:
                return None
            
            r, p_value = self._anosim(valid_distances, binned, permutations)
            
            return {
                'type': 'continuous_binned',
                'method': 'anosim',
                'statistic': r,
                'p_value': p_value,
                'n_samples': len(valid_values),
                'n_bins': n_bins
            }
    
    def _analyze_categorical_variable(self, variable: pd.Series, distances: np.ndarray,
                                    method: str, permutations: int) -> Dict[str, float]:
        """Analyze correlation for categorical variables."""
        # Remove NaN values
        valid_mask = ~variable.isna()
        if valid_mask.sum() < 3:
            logger.warning(f"Insufficient valid values for variable '{variable.name}'")
            return None
        
        valid_indices = np.where(valid_mask)[0]
        valid_values = variable[valid_mask]
        valid_distances = distances[np.ix_(valid_indices, valid_indices)]
        
        # Encode categories
        le = LabelEncoder()
        encoded_values = le.fit_transform(valid_values)
        
        # Check if we have at least 2 categories
        unique_categories = np.unique(encoded_values)
        if len(unique_categories) < 2:
            logger.warning(f"Variable '{variable.name}' has only one category")
            return None
        
        # Check minimum samples per category
        min_samples_per_category = 2
        category_counts = pd.Series(encoded_values).value_counts()
        if (category_counts < min_samples_per_category).any():
            logger.warning(f"Variable '{variable.name}' has categories with too few samples")
            return None
        
        if method == 'anosim':
            r, p_value = self._anosim(valid_distances, encoded_values, permutations)
            
            return {
                'type': 'categorical',
                'method': 'anosim',
                'statistic': r,
                'p_value': p_value,
                'n_samples': len(valid_values),
                'n_categories': len(unique_categories),
                'categories': list(le.classes_)
            }
        else:
            # For Mantel test, create binary distance matrix
            category_distances = np.zeros_like(valid_distances)
            for i in range(len(encoded_values)):
                for j in range(i + 1, len(encoded_values)):
                    if encoded_values[i] != encoded_values[j]:
                        category_distances[i, j] = 1
                        category_distances[j, i] = 1
            
            r, p_value = self._mantel_test(valid_distances, category_distances, permutations)
            
            return {
                'type': 'categorical',
                'method': 'mantel',
                'statistic': r,
                'p_value': p_value,
                'n_samples': len(valid_values),
                'n_categories': len(unique_categories),
                'categories': list(le.classes_)
            }
    
    def _mantel_test(self, dist1: np.ndarray, dist2: np.ndarray, 
                     permutations: int) -> Tuple[float, float]:
        """Perform Mantel test between two distance matrices."""
        # Convert to condensed form
        condensed1 = squareform(dist1)
        condensed2 = squareform(dist2)
        
        # Calculate observed correlation
        observed_r = stats.pearsonr(condensed1, condensed2)[0]
        
        # Permutation test
        permuted_r = []
        n = dist1.shape[0]
        
        for _ in range(permutations):
            # Permute one matrix
            perm = np.random.permutation(n)
            perm_dist = dist2[np.ix_(perm, perm)]
            perm_condensed = squareform(perm_dist)
            
            # Calculate correlation
            r = stats.pearsonr(condensed1, perm_condensed)[0]
            permuted_r.append(r)
        
        # Calculate p-value
        permuted_r = np.array(permuted_r)
        p_value = np.sum(np.abs(permuted_r) >= np.abs(observed_r)) / permutations
        
        return observed_r, p_value
    
    def _anosim(self, distances: np.ndarray, groups: np.ndarray, 
                permutations: int) -> Tuple[float, float]:
        """Perform ANOSIM (Analysis of Similarities) test."""
        n = len(groups)
        unique_groups = np.unique(groups)
        
        # Calculate observed R statistic
        observed_r = self._calculate_anosim_r(distances, groups)
        
        # Permutation test
        permuted_r = []
        
        for _ in range(permutations):
            # Permute group labels
            perm_groups = np.random.permutation(groups)
            r = self._calculate_anosim_r(distances, perm_groups)
            permuted_r.append(r)
        
        # Calculate p-value
        permuted_r = np.array(permuted_r)
        p_value = np.sum(permuted_r >= observed_r) / permutations
        
        return observed_r, p_value
    
    def _calculate_anosim_r(self, distances: np.ndarray, groups: np.ndarray) -> float:
        """Calculate ANOSIM R statistic."""
        n = len(groups)
        
        # Separate within and between group distances
        within_distances = []
        between_distances = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if groups[i] == groups[j]:
                    within_distances.append(distances[i, j])
                else:
                    between_distances.append(distances[i, j])
        
        if not within_distances or not between_distances:
            return 0.0
        
        # Calculate mean ranks
        all_distances = within_distances + between_distances
        ranks = stats.rankdata(all_distances)
        
        within_ranks = ranks[:len(within_distances)]
        between_ranks = ranks[len(within_distances):]
        
        mean_within = np.mean(within_ranks)
        mean_between = np.mean(between_ranks)
        
        # Calculate R statistic
        n_total = len(all_distances)
        r = (mean_between - mean_within) / (n_total / 2)
        
        return r
    
    def identify_significant_variables(self, results: Dict[str, Dict[str, float]], 
                                     alpha: float = 0.05) -> List[Tuple[str, Dict[str, float]]]:
        """Identify metadata variables with significant correlations.
        
        Args:
            results: Correlation analysis results
            alpha: Significance level
            
        Returns:
            List of significant variables sorted by p-value
        """
        significant = []
        
        for var_name, var_results in results.items():
            if var_results['p_value'] <= alpha:
                significant.append((var_name, var_results))
        
        # Sort by p-value
        significant.sort(key=lambda x: x[1]['p_value'])
        
        return significant