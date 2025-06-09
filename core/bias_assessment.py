"""Assessment of bias impact and correction effectiveness."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class BiasAssessment:
    """Assess the impact of technical biases and effectiveness of corrections.
    
    This class provides methods to quantify bias contributions, evaluate
    correction effectiveness, and identify potential confounding factors.
    """
    
    def __init__(self):
        """Initialize the bias assessor."""
        self.assessment_results = {}
        
    def assess_technical_variance(self, data: Union[np.ndarray, pd.DataFrame],
                                 metadata: pd.DataFrame,
                                 technical_vars: List[str],
                                 biological_vars: Optional[List[str]] = None,
                                 n_permutations: int = 999) -> Dict[str, Dict[str, float]]:
        """Assess the contribution of technical variables to total variance.
        
        Uses PERMANOVA-like approach to partition variance components.
        
        Args:
            data: Count matrix (samples x features) or distance matrix
            metadata: Sample metadata
            technical_vars: List of technical variable names
            biological_vars: List of biological variable names
            n_permutations: Number of permutations for significance testing
            
        Returns:
            Dictionary of variance partitioning results
        """
        if isinstance(data, pd.DataFrame):
            data_array = data.values
            sample_names = list(data.index)
        else:
            data_array = data.copy()
            sample_names = [f"Sample_{i}" for i in range(data.shape[0])]
            
        # Align metadata with data
        common_samples = [s for s in sample_names if s in metadata.index]
        aligned_metadata = metadata.loc[common_samples]
        sample_indices = [sample_names.index(s) for s in common_samples]
        aligned_data = data_array[sample_indices, :]
        
        # Calculate distance matrix if not provided
        if aligned_data.shape[0] == aligned_data.shape[1] and np.allclose(aligned_data, aligned_data.T):
            # Assume it's already a distance matrix
            distance_matrix = aligned_data
        else:
            # Calculate Bray-Curtis distances
            distance_matrix = self._calculate_bray_curtis_matrix(aligned_data)
            
        variance_results = {}
        
        # Test each technical variable
        all_vars = technical_vars + (biological_vars or [])
        
        for var_name in all_vars:
            if var_name not in aligned_metadata.columns:
                logger.warning(f"Variable '{var_name}' not found in metadata")
                continue
                
            var_results = self._permanova_test(
                distance_matrix, aligned_metadata, var_name, n_permutations
            )
            
            var_results['variable_type'] = 'technical' if var_name in technical_vars else 'biological'
            variance_results[var_name] = var_results
            
        # Calculate total variance explained by all technical variables
        tech_vars_present = [v for v in technical_vars if v in variance_results]
        if tech_vars_present:
            total_tech_variance = sum(variance_results[v]['variance_explained'] 
                                    for v in tech_vars_present)
            variance_results['_summary'] = {
                'total_technical_variance': total_tech_variance,
                'n_technical_vars': len(tech_vars_present)
            }
            
        self.assessment_results['variance_partitioning'] = variance_results
        
        logger.info(f"Variance assessment complete. Technical variables explain "
                   f"{total_tech_variance:.1%} of variance")
        
        return variance_results
    
    def detect_confounding_factors(self, similarity_matrix: Union[np.ndarray, pd.DataFrame], 
                                 metadata_df: pd.DataFrame,
                                 technical_variables: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """Detect potential confounding factors in sample groupings.
        
        Args:
            similarity_matrix: Sample similarity matrix
            metadata_df: Sample metadata
            technical_variables: Known technical variables to check
            
        Returns:
            Variables ranked by potential confounding effect
        """
        if isinstance(similarity_matrix, pd.DataFrame):
            sample_names = list(similarity_matrix.index)
            sim_matrix = similarity_matrix.values
        else:
            sample_names = [f"Sample_{i}" for i in range(similarity_matrix.shape[0])]
            sim_matrix = similarity_matrix.copy()
            
        # Align metadata with similarity matrix
        common_samples = [s for s in sample_names if s in metadata_df.index]
        aligned_metadata = metadata_df.loc[common_samples]
        sample_indices = [sample_names.index(s) for s in common_samples]
        aligned_sim_matrix = sim_matrix[np.ix_(sample_indices, sample_indices)]
        
        confounding_results = {}
        
        # Auto-detect technical variables if not provided
        if technical_variables is None:
            technical_variables = self._auto_detect_technical_variables(aligned_metadata)
            
        # Test each metadata variable for confounding with similarity structure
        for var_name in aligned_metadata.columns:
            if var_name in technical_variables:
                var_type = 'technical'
            else:
                var_type = 'biological'
                
            # Calculate association between variable and similarity structure
            association_results = self._test_similarity_association(
                aligned_sim_matrix, aligned_metadata[var_name]
            )
            
            association_results['variable_type'] = var_type
            association_results['confounding_risk'] = self._assess_confounding_risk(
                association_results, var_type
            )
            
            confounding_results[var_name] = association_results
            
        # Rank variables by confounding potential
        ranked_results = self._rank_confounding_factors(confounding_results)
        
        # Store results
        self.assessment_results['confounding_factors'] = ranked_results
        
        logger.info(f"Confounding factor analysis complete. {len(ranked_results)} variables tested")
        
        return ranked_results
    
    def _auto_detect_technical_variables(self, metadata: pd.DataFrame) -> List[str]:
        """Auto-detect likely technical variables based on naming patterns.
        
        Args:
            metadata: Sample metadata
            
        Returns:
            List of likely technical variable names
        """
        technical_patterns = [
            'batch', 'run', 'lane', 'extraction', 'library', 'prep', 'kit',
            'sequenc', 'depth', 'reads', 'date', 'technician', 'plate',
            'barcode', 'primer', 'adapter', 'flow_cell', 'machine'
        ]
        
        technical_vars = []
        
        for col in metadata.columns:
            col_lower = col.lower()
            for pattern in technical_patterns:
                if pattern in col_lower:
                    technical_vars.append(col)
                    break
                    
        # Also check for variables with many categories relative to sample size
        for col in metadata.columns:
            if col not in technical_vars:
                if metadata[col].dtype == 'object' or metadata[col].dtype.name == 'category':
                    n_categories = metadata[col].nunique()
                    n_samples = len(metadata)
                    if n_categories > n_samples / 4:  # Many categories per sample
                        technical_vars.append(col)
                        
        logger.info(f"Auto-detected {len(technical_vars)} potential technical variables")
        
        return technical_vars
    
    def _test_similarity_association(self, similarity_matrix: np.ndarray, 
                                   variable: pd.Series) -> Dict[str, float]:
        """Test association between similarity structure and a metadata variable.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            variable: Metadata variable values
            
        Returns:
            Association test results
        """
        # Convert similarity matrix to distance for PERMANOVA
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)
        
        # Create temporary metadata DataFrame for PERMANOVA
        temp_metadata = pd.DataFrame({variable.name: variable})
        
        # Run PERMANOVA test
        permanova_results = self._permanova_test(
            distance_matrix, temp_metadata, variable.name, n_permutations=199
        )
        
        # Additional association tests
        association_results = permanova_results.copy()
        
        # Test correlation with first principal coordinate if continuous
        if pd.api.types.is_numeric_dtype(variable):
            # Perform MDS to get coordinates
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            coords = mds.fit_transform(distance_matrix)
            
            # Test correlation with first coordinate
            correlation, p_corr = stats.pearsonr(variable.values, coords[:, 0])
            association_results['coordinate_correlation'] = abs(correlation)
            association_results['correlation_p_value'] = p_corr
        else:
            association_results['coordinate_correlation'] = 0.0
            association_results['correlation_p_value'] = 1.0
            
        return association_results
    
    def _assess_confounding_risk(self, association_results: Dict[str, float], 
                               variable_type: str) -> str:
        """Assess confounding risk level based on association strength.
        
        Args:
            association_results: Results from association tests
            variable_type: 'technical' or 'biological'
            
        Returns:
            Risk level ('low', 'medium', 'high')
        """
        variance_explained = association_results.get('variance_explained', 0)
        p_value = association_results.get('p_value', 1.0)
        
        # High risk criteria
        if variable_type == 'technical':
            if variance_explained > 0.1 and p_value < 0.05:
                return 'high'
            elif variance_explained > 0.05 and p_value < 0.1:
                return 'medium'
        else:  # biological
            if variance_explained > 0.2 and p_value < 0.01:
                return 'high'
            elif variance_explained > 0.1 and p_value < 0.05:
                return 'medium'
                
        return 'low'
    
    def _rank_confounding_factors(self, confounding_results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Rank variables by confounding potential.
        
        Args:
            confounding_results: Results from confounding analysis
            
        Returns:
            Results sorted by confounding risk
        """
        # Calculate confounding score for ranking
        for var_name, results in confounding_results.items():
            variance_explained = results.get('variance_explained', 0)
            p_value = results.get('p_value', 1.0)
            variable_type = results.get('variable_type', 'biological')
            
            # Score combines effect size and significance
            if p_value > 0:
                score = variance_explained * (-np.log10(p_value))
            else:
                score = variance_explained * 10  # Very significant
                
            # Weight technical variables higher for confounding risk
            if variable_type == 'technical':
                score *= 1.5
                
            results['confounding_score'] = score
            
        # Sort by score
        ranked_results = dict(sorted(
            confounding_results.items(),
            key=lambda x: x[1]['confounding_score'],
            reverse=True
        ))
        
        return ranked_results
    
    def _calculate_bray_curtis_matrix(self, data: np.ndarray) -> np.ndarray:
        """Calculate Bray-Curtis distance matrix."""
        n_samples = data.shape[0]
        dist_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                u = data[i, :]
                v = data[j, :]
                
                numerator = np.sum(np.abs(u - v))
                denominator = np.sum(u) + np.sum(v)
                
                if denominator > 0:
                    bc_dist = numerator / denominator
                else:
                    bc_dist = 0.0
                    
                dist_matrix[i, j] = bc_dist
                dist_matrix[j, i] = bc_dist
                
        return dist_matrix
    
    def _permanova_test(self, distance_matrix: np.ndarray,
                       metadata: pd.DataFrame,
                       variable: str,
                       n_permutations: int = 999) -> Dict[str, float]:
        """Perform PERMANOVA test for a single variable."""
        
        # Remove samples with missing values
        valid_mask = ~metadata[variable].isna()
        if valid_mask.sum() < 3:
            return {'variance_explained': 0.0, 'f_statistic': 0.0, 'p_value': 1.0}
            
        valid_indices = np.where(valid_mask)[0]
        var_distances = distance_matrix[np.ix_(valid_indices, valid_indices)]
        var_metadata = metadata[valid_mask]
        
        # Calculate observed F-statistic
        observed_f = self._calculate_permanova_f(var_distances, var_metadata[variable])
        
        if np.isnan(observed_f):
            return {'variance_explained': 0.0, 'f_statistic': 0.0, 'p_value': 1.0}
            
        # Permutation test
        permuted_f_stats = []
        var_values = var_metadata[variable].values
        
        for _ in range(n_permutations):
            # Permute variable values
            permuted_values = np.random.permutation(var_values)
            permuted_series = pd.Series(permuted_values, index=var_metadata.index)
            
            # Calculate F-statistic for permuted data
            perm_f = self._calculate_permanova_f(var_distances, permuted_series)
            
            if not np.isnan(perm_f):
                permuted_f_stats.append(perm_f)
                
        # Calculate p-value
        if permuted_f_stats:
            p_value = np.sum(np.array(permuted_f_stats) >= observed_f) / len(permuted_f_stats)
        else:
            p_value = 1.0
            
        # Calculate R-squared (variance explained)
        r_squared = self._calculate_permanova_r_squared(var_distances, var_metadata[variable])
        
        return {
            'variance_explained': r_squared,
            'f_statistic': observed_f,
            'p_value': p_value,
            'n_permutations': len(permuted_f_stats)
        }
    
    def _calculate_permanova_f(self, distance_matrix: np.ndarray,
                              grouping_variable: pd.Series) -> float:
        """Calculate PERMANOVA F-statistic."""
        
        n = len(grouping_variable)
        
        # Calculate total sum of squares
        total_ss = np.sum(distance_matrix ** 2) / n
        
        # Calculate within-group and between-group sum of squares
        if grouping_variable.dtype in ['object', 'category'] or grouping_variable.nunique() < 10:
            # Categorical variable
            groups = grouping_variable.unique()
            
            if len(groups) < 2:
                return np.nan
                
            within_ss = 0
            group_sizes = []
            
            for group in groups:
                group_mask = grouping_variable == group
                group_size = np.sum(group_mask)
                
                if group_size < 2:
                    continue
                    
                group_indices = np.where(group_mask)[0]
                group_distances = distance_matrix[np.ix_(group_indices, group_indices)]
                
                # Within-group sum of squares
                within_ss += np.sum(group_distances ** 2) / group_size
                group_sizes.append(group_size)
                
            if len(group_sizes) < 2:
                return np.nan
                
            between_ss = total_ss - within_ss
            
            # Degrees of freedom
            df_between = len(group_sizes) - 1
            df_within = n - len(group_sizes)
            
            if df_within <= 0:
                return np.nan
                
            # F-statistic
            f_stat = (between_ss / df_between) / (within_ss / df_within)
            
        else:
            # Continuous variable - use correlation-based approach
            # This is a simplified version for continuous variables
            correlations = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    dist_ij = distance_matrix[i, j]
                    var_diff = abs(grouping_variable.iloc[i] - grouping_variable.iloc[j])
                    correlations.append((dist_ij, var_diff))
                    
            if len(correlations) < 3:
                return np.nan
                
            distances, var_diffs = zip(*correlations)
            corr, p_val = stats.pearsonr(distances, var_diffs)
            
            # Convert correlation to F-statistic approximation
            n_pairs = len(correlations)
            if abs(corr) < 1:
                f_stat = (corr ** 2) / (1 - corr ** 2) * (n_pairs - 2)
            else:
                f_stat = np.inf
                
        return f_stat
    
    def _calculate_permanova_r_squared(self, distance_matrix: np.ndarray,
                                     grouping_variable: pd.Series) -> float:
        """Calculate R-squared (proportion of variance explained)."""
        
        n = len(grouping_variable)
        
        # Total sum of squares
        total_ss = np.sum(distance_matrix ** 2) / n
        
        if total_ss == 0:
            return 0.0
            
        # Calculate within-group sum of squares
        if grouping_variable.dtype in ['object', 'category'] or grouping_variable.nunique() < 10:
            # Categorical variable
            groups = grouping_variable.unique()
            within_ss = 0
            
            for group in groups:
                group_mask = grouping_variable == group
                group_size = np.sum(group_mask)
                
                if group_size < 2:
                    continue
                    
                group_indices = np.where(group_mask)[0]
                group_distances = distance_matrix[np.ix_(group_indices, group_indices)]
                within_ss += np.sum(group_distances ** 2) / group_size
                
            between_ss = total_ss - within_ss
            r_squared = between_ss / total_ss
            
        else:
            # For continuous variables, use correlation-based R-squared
            correlations = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    dist_ij = distance_matrix[i, j]
                    var_diff = abs(grouping_variable.iloc[i] - grouping_variable.iloc[j])
                    correlations.append((dist_ij, var_diff))
                    
            if len(correlations) < 3:
                return 0.0
                
            distances, var_diffs = zip(*correlations)
            corr, _ = stats.pearsonr(distances, var_diffs)
            r_squared = corr ** 2
            
        return max(0.0, min(1.0, r_squared))
    
    def compare_before_after_correction(self, 
                                      original_data: Union[np.ndarray, pd.DataFrame],
                                      corrected_data: Union[np.ndarray, pd.DataFrame],
                                      metadata: pd.DataFrame,
                                      technical_vars: List[str],
                                      biological_vars: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """Compare data before and after bias correction.
        
        Args:
            original_data: Original data matrix
            corrected_data: Bias-corrected data matrix
            metadata: Sample metadata
            technical_vars: Technical variables that were corrected
            biological_vars: Biological variables to preserve
            
        Returns:
            Comparison metrics
        """
        comparison_results = {}
        
        # Assess technical variance before and after
        original_variance = self.assess_technical_variance(
            original_data, metadata, technical_vars, biological_vars, n_permutations=199
        )
        
        corrected_variance = self.assess_technical_variance(
            corrected_data, metadata, technical_vars, biological_vars, n_permutations=199
        )
        
        # Calculate reduction in technical variance
        for var_name in technical_vars:
            if var_name in original_variance and var_name in corrected_variance:
                original_var = original_variance[var_name]['variance_explained']
                corrected_var = corrected_variance[var_name]['variance_explained']
                
                reduction = original_var - corrected_var
                relative_reduction = reduction / (original_var + 1e-10)
                
                comparison_results[var_name] = {
                    'original_variance': original_var,
                    'corrected_variance': corrected_var,
                    'absolute_reduction': reduction,
                    'relative_reduction': relative_reduction,
                    'original_p_value': original_variance[var_name]['p_value'],
                    'corrected_p_value': corrected_variance[var_name]['p_value']
                }
                
        # Assess preservation of biological signal
        if biological_vars:
            bio_preservation = {}
            
            for var_name in biological_vars:
                if var_name in original_variance and var_name in corrected_variance:
                    original_var = original_variance[var_name]['variance_explained']
                    corrected_var = corrected_variance[var_name]['variance_explained']
                    
                    preservation_ratio = corrected_var / (original_var + 1e-10)
                    
                    bio_preservation[var_name] = {
                        'original_variance': original_var,
                        'corrected_variance': corrected_var,
                        'preservation_ratio': preservation_ratio
                    }
                    
            comparison_results['biological_preservation'] = bio_preservation
            
        # Calculate overall metrics
        tech_vars_present = [v for v in technical_vars if v in comparison_results]
        if tech_vars_present:
            overall_original = np.mean([comparison_results[v]['original_variance'] 
                                      for v in tech_vars_present])
            overall_corrected = np.mean([comparison_results[v]['corrected_variance'] 
                                       for v in tech_vars_present])
            
            comparison_results['_summary'] = {
                'overall_original_variance': overall_original,
                'overall_corrected_variance': overall_corrected,
                'overall_reduction': overall_original - overall_corrected,
                'correction_effectiveness': (overall_original - overall_corrected) / (overall_original + 1e-10)
            }
            
        self.assessment_results['correction_comparison'] = comparison_results
        
        logger.info(f"Correction effectiveness: "
                   f"{comparison_results.get('_summary', {}).get('correction_effectiveness', 0):.1%}")
        
        return comparison_results
    
    def assess_confounding_risk(self, metadata: pd.DataFrame,
                              technical_vars: List[str],
                              biological_vars: List[str]) -> Dict[str, Dict[str, float]]:
        """Assess risk of confounding between technical and biological variables.
        
        Args:
            metadata: Sample metadata
            technical_vars: Technical variable names
            biological_vars: Biological variable names
            
        Returns:
            Confounding assessment results
        """
        confounding_results = {}
        
        for tech_var in technical_vars:
            if tech_var not in metadata.columns:
                continue
                
            tech_confounding = {}
            
            for bio_var in biological_vars:
                if bio_var not in metadata.columns:
                    continue
                    
                # Test association between technical and biological variables
                association = self._test_variable_association(
                    metadata[tech_var], metadata[bio_var]
                )
                
                tech_confounding[bio_var] = association
                
            confounding_results[tech_var] = tech_confounding
            
        # Identify high-risk confounding
        high_risk_pairs = []
        
        for tech_var, bio_associations in confounding_results.items():
            for bio_var, association in bio_associations.items():
                if association['p_value'] < 0.05 and association['effect_size'] > 0.3:
                    high_risk_pairs.append((tech_var, bio_var, association))
                    
        confounding_results['_high_risk_pairs'] = high_risk_pairs
        
        self.assessment_results['confounding_assessment'] = confounding_results
        
        if high_risk_pairs:
            logger.warning(f"High confounding risk detected for {len(high_risk_pairs)} variable pairs")
        else:
            logger.info("No high-risk confounding detected")
            
        return confounding_results
    
    def _test_variable_association(self, var1: pd.Series, var2: pd.Series) -> Dict[str, float]:
        """Test association between two variables."""
        
        # Remove missing values
        valid_mask = ~var1.isna() & ~var2.isna()
        if valid_mask.sum() < 3:
            return {'effect_size': 0.0, 'p_value': 1.0, 'test_type': 'insufficient_data'}
            
        var1_valid = var1[valid_mask]
        var2_valid = var2[valid_mask]
        
        # Determine variable types
        var1_categorical = var1_valid.dtype in ['object', 'category'] or var1_valid.nunique() < 10
        var2_categorical = var2_valid.dtype in ['object', 'category'] or var2_valid.nunique() < 10
        
        if var1_categorical and var2_categorical:
            # Both categorical - Chi-squared test
            try:
                crosstab = pd.crosstab(var1_valid, var2_valid)
                chi2, p_val, _, _ = stats.chi2_contingency(crosstab)
                
                # Cramér's V as effect size
                n = crosstab.sum().sum()
                min_dim = min(crosstab.shape) - 1
                cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                
                return {
                    'effect_size': cramers_v,
                    'p_value': p_val,
                    'test_type': 'chi_squared',
                    'statistic': chi2
                }
            except:
                return {'effect_size': 0.0, 'p_value': 1.0, 'test_type': 'chi_squared_failed'}
                
        elif not var1_categorical and not var2_categorical:
            # Both continuous - Pearson correlation
            corr, p_val = stats.pearsonr(var1_valid, var2_valid)
            
            return {
                'effect_size': abs(corr),
                'p_value': p_val,
                'test_type': 'pearson_correlation',
                'statistic': corr
            }
            
        else:
            # Mixed types - ANOVA
            if var1_categorical:
                categorical_var = var1_valid
                continuous_var = var2_valid
            else:
                categorical_var = var2_valid
                continuous_var = var1_valid
                
            try:
                groups = [continuous_var[categorical_var == cat].values 
                         for cat in categorical_var.unique()]
                groups = [g for g in groups if len(g) > 0]
                
                if len(groups) >= 2:
                    f_stat, p_val = stats.f_oneway(*groups)
                    
                    # Eta-squared as effect size
                    total_var = np.var(continuous_var)
                    if total_var > 0:
                        between_var = np.sum([len(g) * (np.mean(g) - np.mean(continuous_var)) ** 2 
                                            for g in groups]) / len(continuous_var)
                        eta_squared = between_var / total_var
                    else:
                        eta_squared = 0
                        
                    return {
                        'effect_size': eta_squared,
                        'p_value': p_val,
                        'test_type': 'anova',
                        'statistic': f_stat
                    }
                else:
                    return {'effect_size': 0.0, 'p_value': 1.0, 'test_type': 'anova_failed'}
            except:
                return {'effect_size': 0.0, 'p_value': 1.0, 'test_type': 'anova_failed'}
    
    def generate_bias_assessment_report(self) -> Dict[str, Union[str, Dict, List]]:
        """Generate comprehensive bias assessment report.
        
        Returns:
            Formatted report dictionary
        """
        report = {
            'summary': {},
            'technical_variance_assessment': {},
            'correction_effectiveness': {},
            'confounding_assessment': {},
            'recommendations': []
        }
        
        # Summary statistics
        if 'variance_partitioning' in self.assessment_results:
            variance_data = self.assessment_results['variance_partitioning']
            
            if '_summary' in variance_data:
                total_tech_var = variance_data['_summary']['total_technical_variance']
                report['summary']['total_technical_variance'] = f"{total_tech_var:.1%}"
                
                if total_tech_var > 0.2:
                    report['recommendations'].append(
                        "High technical variance detected (>20%). Consider bias correction."
                    )
                elif total_tech_var > 0.1:
                    report['recommendations'].append(
                        "Moderate technical variance detected (>10%). Bias correction recommended."
                    )
                else:
                    report['recommendations'].append(
                        "Low technical variance detected (<10%). Bias correction may not be necessary."
                    )
                    
        # Correction effectiveness
        if 'correction_comparison' in self.assessment_results:
            correction_data = self.assessment_results['correction_comparison']
            
            if '_summary' in correction_data:
                effectiveness = correction_data['_summary']['correction_effectiveness']
                report['summary']['correction_effectiveness'] = f"{effectiveness:.1%}"
                
                if effectiveness > 0.5:
                    report['recommendations'].append(
                        "Bias correction was highly effective (>50% variance reduction)."
                    )
                elif effectiveness > 0.2:
                    report['recommendations'].append(
                        "Bias correction was moderately effective (>20% variance reduction)."
                    )
                else:
                    report['recommendations'].append(
                        "Bias correction had limited effectiveness (<20% variance reduction)."
                    )
                    
        # Confounding assessment
        if 'confounding_assessment' in self.assessment_results:
            confounding_data = self.assessment_results['confounding_assessment']
            
            if '_high_risk_pairs' in confounding_data:
                high_risk_pairs = confounding_data['_high_risk_pairs']
                report['summary']['high_risk_confounding_pairs'] = len(high_risk_pairs)
                
                if high_risk_pairs:
                    pair_names = [f"{pair[0]} vs {pair[1]}" for pair in high_risk_pairs]
                    report['recommendations'].append(
                        f"High confounding risk detected between: {', '.join(pair_names)}. "
                        "Interpret grouping results with caution."
                    )
                else:
                    report['recommendations'].append(
                        "No high-risk confounding detected between technical and biological variables."
                    )
                    
        # Detailed assessments
        report['technical_variance_assessment'] = self.assessment_results.get('variance_partitioning', {})
        report['correction_effectiveness'] = self.assessment_results.get('correction_comparison', {})
        report['confounding_assessment'] = self.assessment_results.get('confounding_assessment', {})
        
        return report