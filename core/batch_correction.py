"""Batch effect correction methods for metagenomic sample data."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


class BatchCorrector:
    """Correct batch effects while preserving biological variation.
    
    This class implements methods to identify and remove technical variation
    from metagenomic data while preserving biological signal.
    """
    
    def __init__(self):
        """Initialize the batch corrector."""
        self.correction_applied = False
        self.batch_effects_detected = {}
        self.correction_method = None
        
    def detect_batch_effects(self, data: Union[np.ndarray, pd.DataFrame],
                           metadata: pd.DataFrame,
                           technical_vars: List[str],
                           biological_vars: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """Detect potential batch effects in the data.
        
        Args:
            data: Count matrix (samples x features) or distance matrix
            metadata: Sample metadata
            technical_vars: List of technical covariate column names
            biological_vars: List of biological variable column names
            
        Returns:
            Dictionary of batch effect statistics for each technical variable
        """
        if isinstance(data, pd.DataFrame):
            data_array = data.values
            sample_names = list(data.index)
        else:
            data_array = data.copy()
            sample_names = [f"Sample_{i}" for i in range(data.shape[0])]
            
        # Align metadata with data
        common_samples = [s for s in sample_names if s in metadata.index]
        if len(common_samples) != len(sample_names):
            logger.warning(f"Only {len(common_samples)}/{len(sample_names)} samples found in metadata")
            
        aligned_metadata = metadata.loc[common_samples]
        sample_indices = [sample_names.index(s) for s in common_samples]
        aligned_data = data_array[sample_indices, :]
        
        batch_effects = {}
        
        for tech_var in technical_vars:
            if tech_var not in aligned_metadata.columns:
                logger.warning(f"Technical variable '{tech_var}' not found in metadata")
                continue
                
            # Remove samples with missing values for this variable
            valid_mask = ~aligned_metadata[tech_var].isna()
            if valid_mask.sum() < 3:
                logger.warning(f"Too few valid values for {tech_var}")
                continue
                
            var_data = aligned_data[valid_mask, :]
            var_metadata = aligned_metadata[valid_mask]
            
            # Calculate batch effect statistics
            effect_stats = self._calculate_batch_effect_stats(
                var_data, var_metadata, tech_var, biological_vars
            )
            
            batch_effects[tech_var] = effect_stats
            
        self.batch_effects_detected = batch_effects
        
        # Log detected effects
        significant_effects = [var for var, stats in batch_effects.items() 
                             if stats.get('p_value', 1.0) < 0.05]
        
        if significant_effects:
            logger.warning(f"Significant batch effects detected for: {', '.join(significant_effects)}")
        else:
            logger.info("No significant batch effects detected")
            
        return batch_effects
    
    def _calculate_batch_effect_stats(self, data: np.ndarray,
                                    metadata: pd.DataFrame,
                                    technical_var: str,
                                    biological_vars: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate statistics for a single technical variable."""
        
        stats_dict = {}
        
        # Get technical variable values
        tech_values = metadata[technical_var]
        
        # Determine if categorical or continuous
        if tech_values.dtype in ['object', 'category'] or tech_values.nunique() < 10:
            # Categorical variable - use PERMANOVA-like approach
            stats_dict.update(self._categorical_batch_stats(data, tech_values))
        else:
            # Continuous variable - use correlation approach
            stats_dict.update(self._continuous_batch_stats(data, tech_values))
            
        # Calculate variance explained
        stats_dict['variance_explained'] = self._calculate_variance_explained(
            data, tech_values
        )
        
        # Check confounding with biological variables
        if biological_vars:
            confounding_stats = self._check_confounding(
                metadata, technical_var, biological_vars
            )
            stats_dict.update(confounding_stats)
            
        return stats_dict
    
    def _categorical_batch_stats(self, data: np.ndarray,
                               categorical_var: pd.Series) -> Dict[str, float]:
        """Calculate batch effect statistics for categorical variables."""
        
        # Encode categories
        le = LabelEncoder()
        encoded_labels = le.fit_transform(categorical_var.fillna('Missing'))
        unique_labels = np.unique(encoded_labels)
        
        if len(unique_labels) < 2:
            return {'p_value': 1.0, 'f_statistic': 0.0}
            
        # Perform multivariate ANOVA (MANOVA) on PC scores
        # First, reduce dimensionality
        n_components = min(10, data.shape[0] - 1, data.shape[1])
        if n_components < 2:
            return {'p_value': 1.0, 'f_statistic': 0.0}
            
        pca = PCA(n_components=n_components)
        pc_scores = pca.fit_transform(data)
        
        # Calculate F-statistics for each PC
        f_statistics = []
        p_values = []
        
        for i in range(n_components):
            # One-way ANOVA for this PC
            groups = [pc_scores[encoded_labels == label, i] for label in unique_labels]
            groups = [g for g in groups if len(g) > 0]  # Remove empty groups
            
            if len(groups) >= 2:
                f_stat, p_val = stats.f_oneway(*groups)
                f_statistics.append(f_stat)
                p_values.append(p_val)
                
        if f_statistics:
            # Combine p-values using Fisher's method
            combined_chi2 = -2 * np.sum(np.log(np.array(p_values) + 1e-10))
            combined_p = stats.chi2.sf(combined_chi2, 2 * len(p_values))
            
            return {
                'f_statistic': np.mean(f_statistics),
                'p_value': combined_p,
                'n_categories': len(unique_labels)
            }
        else:
            return {'p_value': 1.0, 'f_statistic': 0.0}
    
    def _continuous_batch_stats(self, data: np.ndarray,
                              continuous_var: pd.Series) -> Dict[str, float]:
        """Calculate batch effect statistics for continuous variables."""
        
        # Remove missing values
        valid_mask = ~continuous_var.isna()
        if valid_mask.sum() < 3:
            return {'p_value': 1.0, 'correlation': 0.0}
            
        var_data = data[valid_mask, :]
        var_values = continuous_var[valid_mask].values
        
        # Calculate correlation with each feature
        correlations = []
        p_values = []
        
        for j in range(var_data.shape[1]):
            feature_values = var_data[:, j]
            
            # Skip features with no variation
            if np.std(feature_values) == 0:
                continue
                
            corr, p_val = stats.pearsonr(feature_values, var_values)
            correlations.append(np.abs(corr))  # Use absolute correlation
            p_values.append(p_val)
            
        if correlations:
            # Average correlation and combine p-values
            mean_correlation = np.mean(correlations)
            
            # Fisher's method for combining p-values
            combined_chi2 = -2 * np.sum(np.log(np.array(p_values) + 1e-10))
            combined_p = stats.chi2.sf(combined_chi2, 2 * len(p_values))
            
            return {
                'correlation': mean_correlation,
                'p_value': combined_p
            }
        else:
            return {'p_value': 1.0, 'correlation': 0.0}
    
    def _calculate_variance_explained(self, data: np.ndarray,
                                    covariate: pd.Series) -> float:
        """Calculate proportion of variance explained by covariate."""
        
        # Remove missing values
        valid_mask = ~covariate.isna()
        if valid_mask.sum() < 3:
            return 0.0
            
        var_data = data[valid_mask, :]
        
        # For categorical variables, use ANOVA-like approach
        if covariate.dtype in ['object', 'category'] or covariate.nunique() < 10:
            # Encode categories
            le = LabelEncoder()
            encoded = le.fit_transform(covariate[valid_mask].fillna('Missing'))
            
            # Calculate R-squared for each feature
            r_squared_values = []
            
            for j in range(var_data.shape[1]):
                feature_values = var_data[:, j]
                
                # Skip features with no variation
                if np.std(feature_values) == 0:
                    continue
                    
                # Create dummy variables
                unique_values = np.unique(encoded)
                if len(unique_values) < 2:
                    continue
                    
                # Calculate sum of squares
                total_ss = np.sum((feature_values - np.mean(feature_values)) ** 2)
                
                if total_ss == 0:
                    continue
                    
                # Between-group sum of squares
                between_ss = 0
                for val in unique_values:
                    group_mask = encoded == val
                    if np.sum(group_mask) > 0:
                        group_mean = np.mean(feature_values[group_mask])
                        between_ss += np.sum(group_mask) * (group_mean - np.mean(feature_values)) ** 2
                        
                r_squared = between_ss / total_ss
                r_squared_values.append(r_squared)
                
        else:
            # For continuous variables, use linear regression
            covariate_values = covariate[valid_mask].values.reshape(-1, 1)
            
            r_squared_values = []
            
            for j in range(var_data.shape[1]):
                feature_values = var_data[:, j]
                
                # Skip features with no variation
                if np.std(feature_values) == 0:
                    continue
                    
                # Linear regression
                reg = LinearRegression().fit(covariate_values, feature_values)
                r_squared = reg.score(covariate_values, feature_values)
                r_squared_values.append(max(0, r_squared))  # Ensure non-negative
                
        if r_squared_values:
            return np.mean(r_squared_values)
        else:
            return 0.0
    
    def _check_confounding(self, metadata: pd.DataFrame,
                          technical_var: str,
                          biological_vars: List[str]) -> Dict[str, float]:
        """Check for confounding between technical and biological variables."""
        
        confounding_stats = {}
        
        for bio_var in biological_vars:
            if bio_var not in metadata.columns:
                continue
                
            # Remove samples with missing values
            valid_mask = ~metadata[technical_var].isna() & ~metadata[bio_var].isna()
            if valid_mask.sum() < 3:
                continue
                
            tech_values = metadata.loc[valid_mask, technical_var]
            bio_values = metadata.loc[valid_mask, bio_var]
            
            # Calculate association between technical and biological variables
            if (tech_values.dtype in ['object', 'category'] and 
                bio_values.dtype in ['object', 'category']):
                # Both categorical - use chi-squared test
                try:
                    crosstab = pd.crosstab(tech_values, bio_values)
                    chi2, p_val, _, _ = stats.chi2_contingency(crosstab)
                    confounding_stats[f'confounding_{bio_var}_chi2'] = chi2
                    confounding_stats[f'confounding_{bio_var}_p'] = p_val
                except:
                    pass
                    
            elif (tech_values.dtype not in ['object', 'category'] and 
                  bio_values.dtype not in ['object', 'category']):
                # Both continuous - use correlation
                corr, p_val = stats.pearsonr(tech_values, bio_values)
                confounding_stats[f'confounding_{bio_var}_corr'] = abs(corr)
                confounding_stats[f'confounding_{bio_var}_p'] = p_val
                
            else:
                # Mixed types - use ANOVA
                if tech_values.dtype in ['object', 'category']:
                    categorical_var = tech_values
                    continuous_var = bio_values
                else:
                    categorical_var = bio_values
                    continuous_var = tech_values
                    
                try:
                    groups = [continuous_var[categorical_var == cat].values 
                             for cat in categorical_var.unique()]
                    groups = [g for g in groups if len(g) > 0]
                    
                    if len(groups) >= 2:
                        f_stat, p_val = stats.f_oneway(*groups)
                        confounding_stats[f'confounding_{bio_var}_f'] = f_stat
                        confounding_stats[f'confounding_{bio_var}_p'] = p_val
                except:
                    pass
                    
        return confounding_stats
    
    def apply_combat_correction(self, data: Union[np.ndarray, pd.DataFrame],
                              metadata: pd.DataFrame,
                              batch_var: str,
                              biological_vars: Optional[List[str]] = None,
                              parametric: bool = True) -> Union[np.ndarray, pd.DataFrame]:
        """Apply ComBat batch correction.
        
        This is a simplified implementation of the ComBat algorithm
        for correcting batch effects.
        
        Args:
            data: Count matrix (samples x features)
            metadata: Sample metadata
            batch_var: Batch variable column name
            biological_vars: Biological variables to preserve
            parametric: Use parametric estimation
            
        Returns:
            Batch-corrected data
        """
        if isinstance(data, pd.DataFrame):
            data_array = data.values.T  # ComBat expects features x samples
            sample_names = list(data.index)
            feature_names = list(data.columns)
            return_df = True
        else:
            data_array = data.T  # Transpose for features x samples
            sample_names = [f"Sample_{i}" for i in range(data.shape[0])]
            feature_names = [f"Feature_{i}" for i in range(data.shape[1])]
            return_df = False
            
        # Align metadata
        common_samples = [s for s in sample_names if s in metadata.index]
        aligned_metadata = metadata.loc[common_samples]
        sample_indices = [sample_names.index(s) for s in common_samples]
        aligned_data = data_array[:, sample_indices]
        
        # Get batch information
        batch_info = aligned_metadata[batch_var]
        
        # Remove samples with missing batch information
        valid_mask = ~batch_info.isna()
        if valid_mask.sum() < 3:
            logger.warning("Too few samples with valid batch information")
            return data if not return_df else data
            
        batch_info = batch_info[valid_mask]
        aligned_data = aligned_data[:, valid_mask]
        
        # Encode batches
        le = LabelEncoder()
        batch_encoded = le.fit_transform(batch_info)
        n_batches = len(np.unique(batch_encoded))
        
        if n_batches < 2:
            logger.warning("Less than 2 batches found")
            return data if not return_df else data
            
        # Create design matrix for biological variables
        if biological_vars:
            bio_design = self._create_design_matrix(aligned_metadata[valid_mask], biological_vars)
        else:
            bio_design = np.ones((np.sum(valid_mask), 1))  # Intercept only
            
        # Apply ComBat correction
        corrected_data = self._combat_algorithm(
            aligned_data, batch_encoded, bio_design, parametric
        )
        
        # Reconstruct full data matrix
        full_corrected = data_array.copy()
        valid_indices = np.where(valid_mask)[0]
        full_corrected[:, [sample_indices[i] for i in valid_indices]] = corrected_data
        
        # Transpose back to samples x features
        full_corrected = full_corrected.T
        
        self.correction_applied = True
        self.correction_method = 'combat'
        
        logger.info(f"ComBat correction applied for {n_batches} batches")
        
        if return_df:
            return pd.DataFrame(full_corrected, index=sample_names, columns=feature_names)
        return full_corrected
    
    def _create_design_matrix(self, metadata: pd.DataFrame,
                            variables: List[str]) -> np.ndarray:
        """Create design matrix for biological variables."""
        
        design_columns = []
        
        # Always include intercept
        design_columns.append(np.ones(len(metadata)))
        
        for var in variables:
            if var not in metadata.columns:
                continue
                
            values = metadata[var]
            
            if values.dtype in ['object', 'category']:
                # Categorical variable - create dummy variables
                unique_vals = values.unique()
                for val in unique_vals[1:]:  # Skip first category (reference)
                    dummy = (values == val).astype(float)
                    design_columns.append(dummy.values)
            else:
                # Continuous variable
                # Standardize for numerical stability
                standardized = (values - values.mean()) / (values.std() + 1e-10)
                design_columns.append(standardized.values)
                
        return np.column_stack(design_columns)
    
    def _combat_algorithm(self, data: np.ndarray,
                         batch_labels: np.ndarray,
                         design_matrix: np.ndarray,
                         parametric: bool = True) -> np.ndarray:
        """Core ComBat algorithm implementation."""
        
        n_features, n_samples = data.shape
        n_batches = len(np.unique(batch_labels))
        
        # Standardize data
        data_mean = np.mean(data, axis=1, keepdims=True)
        data_std = np.std(data, axis=1, keepdims=True) + 1e-10
        standardized_data = (data - data_mean) / data_std
        
        # Fit linear model to remove biological effects
        coefficients = np.zeros((n_features, design_matrix.shape[1]))
        residuals = np.zeros_like(standardized_data)
        
        for i in range(n_features):
            try:
                reg = LinearRegression(fit_intercept=False)
                reg.fit(design_matrix, standardized_data[i, :])
                coefficients[i, :] = reg.coef_
                residuals[i, :] = standardized_data[i, :] - reg.predict(design_matrix)
            except:
                # If regression fails, use original data
                residuals[i, :] = standardized_data[i, :]
                
        # Estimate batch effects
        batch_effects = {}
        
        for batch_id in np.unique(batch_labels):
            batch_mask = batch_labels == batch_id
            
            if np.sum(batch_mask) < 2:
                continue
                
            # Additive batch effect (location)
            gamma = np.mean(residuals[:, batch_mask], axis=1)
            
            # Multiplicative batch effect (scale)
            if parametric:
                delta_sq = np.var(residuals[:, batch_mask], axis=1, ddof=1)
            else:
                delta_sq = np.median(np.abs(residuals[:, batch_mask] - 
                                          np.median(residuals[:, batch_mask], axis=1, keepdims=True)), 
                                   axis=1) ** 2
                                   
            batch_effects[batch_id] = {
                'gamma': gamma,
                'delta_sq': delta_sq
            }
            
        # Apply batch correction
        corrected_residuals = residuals.copy()
        
        for batch_id in np.unique(batch_labels):
            batch_mask = batch_labels == batch_id
            
            if batch_id not in batch_effects:
                continue
                
            gamma = batch_effects[batch_id]['gamma']
            delta_sq = batch_effects[batch_id]['delta_sq']
            delta = np.sqrt(np.maximum(delta_sq, 1e-10))
            
            # Correct batch effects
            corrected_residuals[:, batch_mask] = (
                (residuals[:, batch_mask] - gamma[:, np.newaxis]) / delta[:, np.newaxis]
            )
            
        # Add back biological effects
        corrected_data = corrected_residuals.copy()
        
        for i in range(n_features):
            biological_effect = np.dot(design_matrix, coefficients[i, :])
            corrected_data[i, :] += biological_effect
            
        # Unstandardize
        corrected_data = corrected_data * data_std + data_mean
        
        return corrected_data
    
    def apply_linear_correction(self, data: Union[np.ndarray, pd.DataFrame],
                              metadata: pd.DataFrame,
                              technical_vars: List[str],
                              biological_vars: Optional[List[str]] = None) -> Union[np.ndarray, pd.DataFrame]:
        """Apply linear model-based batch correction.
        
        Args:
            data: Count matrix (samples x features)
            metadata: Sample metadata
            technical_vars: Technical variables to correct for
            biological_vars: Biological variables to preserve
            
        Returns:
            Corrected data
        """
        if isinstance(data, pd.DataFrame):
            data_array = data.values
            return_df = True
            df_index = data.index
            df_columns = data.columns
        else:
            data_array = data.copy()
            return_df = False
            
        # Create combined design matrix
        all_vars = (biological_vars or []) + technical_vars
        design_matrix = self._create_design_matrix(metadata, all_vars)
        
        # Identify which columns correspond to technical variables
        n_bio = len(biological_vars) if biological_vars else 0
        n_intercept = 1  # Always have intercept
        tech_start_idx = n_intercept + n_bio
        
        corrected_data = np.zeros_like(data_array)
        
        # Apply correction to each feature
        for j in range(data_array.shape[1]):
            feature_data = data_array[:, j]
            
            # Fit linear model
            try:
                reg = LinearRegression(fit_intercept=False)
                reg.fit(design_matrix, feature_data)
                
                # Predict biological effects only (exclude technical effects)
                bio_design = design_matrix[:, :tech_start_idx]
                bio_coeffs = reg.coef_[:tech_start_idx]
                
                biological_effects = np.dot(bio_design, bio_coeffs)
                
                # Residual after removing all effects
                residual = feature_data - reg.predict(design_matrix)
                
                # Corrected data = biological effects + residual
                corrected_data[:, j] = biological_effects + residual
                
            except:
                # If regression fails, use original data
                corrected_data[:, j] = feature_data
                
        self.correction_applied = True
        self.correction_method = 'linear'
        
        logger.info(f"Linear correction applied for variables: {technical_vars}")
        
        if return_df:
            return pd.DataFrame(corrected_data, index=df_index, columns=df_columns)
        return corrected_data