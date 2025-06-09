"""Optimization of sample groupings for metagenomic assembly."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


class GroupingOptimizer:
    """Optimize sample groupings based on sequence similarity and metadata.
    
    This class provides methods to generate and evaluate different sample
    grouping strategies to maximize within-group homogeneity.
    """
    
    def __init__(self):
        """Initialize the grouping optimizer."""
        self.distance_matrix = None
        self.sample_names = None
        self.metadata = None
        
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
        
    def set_metadata(self, metadata: pd.DataFrame):
        """Set the metadata DataFrame.
        
        Args:
            metadata: DataFrame with samples as index
        """
        self.metadata = metadata
    
    def generate_metadata_groupings(self, variable: str, 
                                  min_samples_per_group: int = 3) -> Dict[str, List[str]]:
        """Generate sample groupings based on a metadata variable.
        
        Args:
            variable: Metadata variable name to use for grouping
            min_samples_per_group: Minimum samples required per group
            
        Returns:
            Dictionary mapping group names to lists of sample names
        """
        if self.metadata is None:
            raise ValueError("Metadata not set")
        
        if variable not in self.metadata.columns:
            raise ValueError(f"Variable '{variable}' not found in metadata")
        
        # Get samples with valid values for this variable
        valid_mask = ~self.metadata[variable].isna()
        valid_samples = self.metadata[valid_mask]
        
        # Group by variable
        groupings = {}
        for group_value, group_df in valid_samples.groupby(variable):
            group_samples = list(group_df.index)
            
            # Only include groups with sufficient samples
            if len(group_samples) >= min_samples_per_group:
                groupings[str(group_value)] = group_samples
            else:
                logger.warning(f"Excluding group '{group_value}' with only {len(group_samples)} samples")
        
        return groupings
    
    def generate_clustering_groupings(self, n_clusters: Union[int, List[int]], 
                                    method: str = 'kmeans') -> Dict[int, Dict[str, List[str]]]:
        """Generate sample groupings using clustering algorithms.
        
        Args:
            n_clusters: Number of clusters or list of cluster numbers to try
            method: Clustering method ('kmeans' or 'hierarchical')
            
        Returns:
            Dictionary mapping cluster numbers to grouping dictionaries
        """
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not set")
        
        if isinstance(n_clusters, int):
            n_clusters = [n_clusters]
        
        results = {}
        
        for k in n_clusters:
            if k >= len(self.sample_names):
                logger.warning(f"Skipping k={k} (>= number of samples)")
                continue
            
            if method == 'kmeans':
                # Convert distance matrix to feature matrix using MDS
                from sklearn.manifold import MDS
                mds = MDS(n_components=min(10, len(self.sample_names) - 1), 
                         dissimilarity='precomputed', random_state=42)
                features = mds.fit_transform(self.distance_matrix)
                
                # Perform k-means clustering
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                
            elif method == 'hierarchical':
                # Perform hierarchical clustering
                clustering = AgglomerativeClustering(
                    n_clusters=k, 
                    metric='precomputed',
                    linkage='average'
                )
                labels = clustering.fit_predict(self.distance_matrix)
                
            else:
                raise ValueError(f"Unknown clustering method: {method}")
            
            # Create groupings
            groupings = {}
            for cluster_id in range(k):
                cluster_samples = [self.sample_names[i] for i, l in enumerate(labels) if l == cluster_id]
                if cluster_samples:  # Only add non-empty clusters
                    groupings[f"cluster_{cluster_id}"] = cluster_samples
            
            results[k] = groupings
        
        return results
    
    def evaluate_grouping(self, grouping: Dict[str, List[str]]) -> Dict[str, float]:
        """Evaluate the quality of a sample grouping.
        
        Args:
            grouping: Dictionary mapping group names to lists of sample names
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not set")
        
        # Get sample indices for each group
        all_samples = []
        labels = []
        
        for group_id, (group_name, samples) in enumerate(grouping.items()):
            for sample in samples:
                if sample in self.sample_names:
                    idx = self.sample_names.index(sample)
                    all_samples.append(idx)
                    labels.append(group_id)
        
        if len(set(labels)) < 2:
            logger.warning("Less than 2 groups found, cannot calculate clustering metrics")
            return {
                'n_groups': len(grouping),
                'n_samples': len(all_samples),
                'silhouette_score': np.nan,
                'within_group_distance': self._calculate_within_group_distance(grouping),
                'between_group_distance': np.nan
            }
        
        # Extract relevant distance matrix
        sample_indices = np.array(all_samples)
        sub_distance_matrix = self.distance_matrix[np.ix_(sample_indices, sample_indices)]
        
        # Calculate metrics
        metrics = {
            'n_groups': len(grouping),
            'n_samples': len(all_samples),
            'silhouette_score': silhouette_score(sub_distance_matrix, labels, metric='precomputed'),
            'within_group_distance': self._calculate_within_group_distance(grouping),
            'between_group_distance': self._calculate_between_group_distance(grouping)
        }
        
        # Add per-group statistics
        group_stats = self._calculate_group_statistics(grouping)
        metrics['group_statistics'] = group_stats
        
        return metrics
    
    def _calculate_within_group_distance(self, grouping: Dict[str, List[str]]) -> float:
        """Calculate average within-group distance."""
        total_distance = 0
        total_pairs = 0
        
        for group_name, samples in grouping.items():
            # Get indices for samples in this group
            indices = []
            for sample in samples:
                if sample in self.sample_names:
                    indices.append(self.sample_names.index(sample))
            
            # Calculate pairwise distances within group
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    total_distance += self.distance_matrix[indices[i], indices[j]]
                    total_pairs += 1
        
        return total_distance / total_pairs if total_pairs > 0 else 0
    
    def _calculate_between_group_distance(self, grouping: Dict[str, List[str]]) -> float:
        """Calculate average between-group distance."""
        total_distance = 0
        total_pairs = 0
        
        group_items = list(grouping.items())
        
        for i in range(len(group_items)):
            group1_name, group1_samples = group_items[i]
            
            # Get indices for group 1
            group1_indices = []
            for sample in group1_samples:
                if sample in self.sample_names:
                    group1_indices.append(self.sample_names.index(sample))
            
            for j in range(i + 1, len(group_items)):
                group2_name, group2_samples = group_items[j]
                
                # Get indices for group 2
                group2_indices = []
                for sample in group2_samples:
                    if sample in self.sample_names:
                        group2_indices.append(self.sample_names.index(sample))
                
                # Calculate distances between groups
                for idx1 in group1_indices:
                    for idx2 in group2_indices:
                        total_distance += self.distance_matrix[idx1, idx2]
                        total_pairs += 1
        
        return total_distance / total_pairs if total_pairs > 0 else 0
    
    def _calculate_group_statistics(self, grouping: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each group."""
        stats = {}
        
        for group_name, samples in grouping.items():
            # Get indices for samples in this group
            indices = []
            for sample in samples:
                if sample in self.sample_names:
                    indices.append(self.sample_names.index(sample))
            
            if len(indices) < 2:
                stats[group_name] = {
                    'n_samples': len(indices),
                    'mean_distance': np.nan,
                    'std_distance': np.nan,
                    'max_distance': np.nan
                }
                continue
            
            # Calculate pairwise distances within group
            distances = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    distances.append(self.distance_matrix[indices[i], indices[j]])
            
            stats[group_name] = {
                'n_samples': len(indices),
                'mean_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'max_distance': np.max(distances)
            }
        
        return stats
    
    def optimize_grouping_number(self, min_clusters: int = 2, max_clusters: int = 10,
                               method: str = 'kmeans') -> Tuple[int, Dict[str, float]]:
        """Find optimal number of groups using clustering.
        
        Args:
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
            method: Clustering method to use
            
        Returns:
            Tuple of (optimal number of clusters, evaluation metrics)
        """
        max_clusters = min(max_clusters, len(self.sample_names) - 1)
        
        if max_clusters < min_clusters:
            raise ValueError("Not enough samples for requested cluster range")
        
        # Generate clusterings for different k values
        clusterings = self.generate_clustering_groupings(
            list(range(min_clusters, max_clusters + 1)), 
            method=method
        )
        
        # Evaluate each clustering
        results = {}
        for k, grouping in clusterings.items():
            metrics = self.evaluate_grouping(grouping)
            results[k] = metrics
        
        # Find optimal k based on silhouette score
        best_k = min_clusters
        best_score = -1
        
        for k, metrics in results.items():
            if not np.isnan(metrics['silhouette_score']) and metrics['silhouette_score'] > best_score:
                best_score = metrics['silhouette_score']
                best_k = k
        
        return best_k, results
    
    def recommend_assembly_strategy(self, grouping_results: Dict[str, Dict[str, float]], 
                                  threshold: float = 0.3) -> Dict[str, Union[str, float, Dict]]:
        """Recommend assembly strategy based on grouping analysis.
        
        Args:
            grouping_results: Evaluation results for different groupings
            threshold: Silhouette score threshold for recommending grouped assembly
            
        Returns:
            Dictionary with recommendation and supporting metrics
        """
        recommendation = {
            'strategy': 'direct',
            'reason': '',
            'confidence': 0.0,
            'metrics': {}
        }
        
        # Find best grouping
        best_grouping = None
        best_score = -1
        
        for name, metrics in grouping_results.items():
            if not np.isnan(metrics.get('silhouette_score', -1)):
                if metrics['silhouette_score'] > best_score:
                    best_score = metrics['silhouette_score']
                    best_grouping = name
        
        if best_grouping and best_score > threshold:
            # Calculate confidence based on various factors
            within_dist = grouping_results[best_grouping]['within_group_distance']
            between_dist = grouping_results[best_grouping]['between_group_distance']
            
            if between_dist > 0:
                separation_ratio = between_dist / within_dist
            else:
                separation_ratio = 1.0
            
            confidence = min(1.0, (best_score + min(1.0, separation_ratio - 1)) / 2)
            
            recommendation['strategy'] = 'hierarchical'
            recommendation['reason'] = f"Clear sample groupings detected with silhouette score {best_score:.3f}"
            recommendation['confidence'] = confidence
            recommendation['best_grouping'] = best_grouping
            recommendation['metrics'] = {
                'silhouette_score': best_score,
                'within_group_distance': within_dist,
                'between_group_distance': between_dist,
                'separation_ratio': separation_ratio
            }
        else:
            recommendation['reason'] = "No clear sample groupings detected"
            recommendation['confidence'] = 0.8  # High confidence in direct assembly
            recommendation['metrics'] = {
                'best_silhouette_score': best_score if best_score > -1 else None
            }
        
        return recommendation