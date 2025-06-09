"""Visualization tools for sample grouping analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import logging

logger = logging.getLogger(__name__)


class SampleGroupingVisualizer:
    """Generate visualizations for sample relationships and groupings.
    
    This class provides methods to create various plots for visualizing
    sequence-based distances, metadata correlations, and sample groupings.
    """
    
    def __init__(self, figure_dir: Optional[Union[str, Path]] = None):
        """Initialize the visualizer.
        
        Args:
            figure_dir: Directory to save figures (optional)
        """
        self.figure_dir = Path(figure_dir) if figure_dir else None
        if self.figure_dir:
            self.figure_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['font.size'] = 12
    
    def plot_distance_heatmap(self, distance_matrix: np.ndarray, sample_names: List[str],
                            grouping: Optional[Dict[str, List[str]]] = None,
                            title: str = "Sample Distance Heatmap",
                            save_name: Optional[str] = None) -> plt.Figure:
        """Create a heatmap of sample distances.
        
        Args:
            distance_matrix: Square distance matrix
            sample_names: List of sample names
            grouping: Optional grouping dictionary for annotation
            title: Plot title
            save_name: Filename to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create DataFrame for easier manipulation
        df_dist = pd.DataFrame(distance_matrix, index=sample_names, columns=sample_names)
        
        # If grouping provided, sort samples by group
        if grouping:
            sorted_samples = []
            group_colors = []
            color_palette = sns.color_palette("husl", len(grouping))
            
            for i, (group_name, samples) in enumerate(grouping.items()):
                group_samples = [s for s in samples if s in sample_names]
                sorted_samples.extend(group_samples)
                group_colors.extend([color_palette[i]] * len(group_samples))
            
            # Add any remaining samples not in groups
            remaining = set(sample_names) - set(sorted_samples)
            sorted_samples.extend(list(remaining))
            group_colors.extend(['gray'] * len(remaining))
            
            # Reorder matrix
            df_dist = df_dist.loc[sorted_samples, sorted_samples]
            
            # Create row colors for groups
            row_colors = pd.Series(group_colors, index=sorted_samples, name='Group')
        else:
            row_colors = None
        
        # Create clustermap
        g = sns.clustermap(df_dist, 
                          cmap='viridis_r',
                          row_cluster=False if grouping else True,
                          col_cluster=False if grouping else True,
                          row_colors=row_colors,
                          col_colors=row_colors,
                          cbar_kws={'label': 'Distance'},
                          figsize=(12, 10))
        
        g.fig.suptitle(title, y=1.02)
        
        if save_name and self.figure_dir:
            filepath = self.figure_dir / save_name
            g.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
        
        return g.fig
    
    def plot_ordination(self, distance_matrix: np.ndarray, sample_names: List[str],
                       method: str = 'MDS', grouping: Optional[Dict[str, List[str]]] = None,
                       metadata: Optional[pd.DataFrame] = None, 
                       color_by: Optional[str] = None,
                       title: Optional[str] = None,
                       save_name: Optional[str] = None) -> plt.Figure:
        """Create ordination plot (PCA/MDS/t-SNE) of samples.
        
        Args:
            distance_matrix: Square distance matrix
            sample_names: List of sample names
            method: Ordination method ('PCA', 'MDS', 't-SNE')
            grouping: Optional grouping dictionary
            metadata: Optional metadata DataFrame
            color_by: Metadata column to use for coloring
            title: Plot title
            save_name: Filename to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Perform ordination
        if method == 'MDS':
            ordination = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            coords = ordination.fit_transform(distance_matrix)
            stress = ordination.stress_
            default_title = f"MDS Plot (Stress: {stress:.3f})"
        elif method == 't-SNE':
            ordination = TSNE(n_components=2, metric='precomputed', random_state=42)
            coords = ordination.fit_transform(distance_matrix)
            default_title = "t-SNE Plot"
        elif method == 'PCA':
            # For PCA, we need features not distances
            # Use MDS to convert distances to features first
            mds = MDS(n_components=min(10, len(sample_names) - 1), 
                     dissimilarity='precomputed', random_state=42)
            features = mds.fit_transform(distance_matrix)
            
            pca = PCA(n_components=2)
            coords = pca.fit_transform(features)
            var_explained = pca.explained_variance_ratio_
            default_title = f"PCA Plot (PC1: {var_explained[0]:.1%}, PC2: {var_explained[1]:.1%})"
        else:
            raise ValueError(f"Unknown ordination method: {method}")
        
        title = title or default_title
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame(coords, columns=['Dim1', 'Dim2'])
        plot_df['Sample'] = sample_names
        
        # Add grouping or metadata coloring
        if grouping:
            # Map samples to groups
            sample_to_group = {}
            for group_name, samples in grouping.items():
                for sample in samples:
                    sample_to_group[sample] = group_name
            
            plot_df['Group'] = plot_df['Sample'].map(sample_to_group).fillna('Ungrouped')
            hue_col = 'Group'
            
        elif metadata is not None and color_by:
            # Map samples to metadata values
            plot_df['Color'] = plot_df['Sample'].map(
                lambda x: metadata.loc[x, color_by] if x in metadata.index else None
            )
            hue_col = 'Color'
        else:
            hue_col = None
        
        # Create scatter plot
        if hue_col:
            # Determine if categorical or continuous
            if plot_df[hue_col].dtype in ['float64', 'int64']:
                # Continuous variable
                scatter = ax.scatter(plot_df['Dim1'], plot_df['Dim2'], 
                                   c=plot_df[hue_col], cmap='viridis', s=100)
                plt.colorbar(scatter, ax=ax, label=hue_col)
            else:
                # Categorical variable
                unique_values = plot_df[hue_col].nunique()
                palette = sns.color_palette("husl", unique_values)
                
                for i, value in enumerate(plot_df[hue_col].unique()):
                    mask = plot_df[hue_col] == value
                    ax.scatter(plot_df.loc[mask, 'Dim1'], 
                             plot_df.loc[mask, 'Dim2'],
                             label=value, s=100, color=palette[i % len(palette)])
                
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter(plot_df['Dim1'], plot_df['Dim2'], s=100)
        
        # Add sample labels if not too many
        if len(sample_names) <= 20:
            for idx, row in plot_df.iterrows():
                ax.annotate(row['Sample'], (row['Dim1'], row['Dim2']), 
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_name and self.figure_dir:
            filepath = self.figure_dir / save_name
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
        
        return fig
    
    def plot_dendrogram(self, distance_matrix: np.ndarray, sample_names: List[str],
                       grouping: Optional[Dict[str, List[str]]] = None,
                       title: str = "Hierarchical Clustering Dendrogram",
                       save_name: Optional[str] = None) -> plt.Figure:
        """Create a hierarchical clustering dendrogram.
        
        Args:
            distance_matrix: Square distance matrix
            sample_names: List of sample names
            grouping: Optional grouping dictionary for coloring
            title: Plot title
            save_name: Filename to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Convert to condensed distance matrix
        condensed_dist = squareform(distance_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_dist, method='average')
        
        # Create color mapping if grouping provided
        if grouping:
            # Map samples to groups
            sample_to_group = {}
            for group_name, samples in grouping.items():
                for sample in samples:
                    sample_to_group[sample] = group_name
            
            # Create color list
            unique_groups = list(grouping.keys()) + ['Ungrouped']
            color_palette = sns.color_palette("husl", len(unique_groups))
            group_to_color = {group: color for group, color in zip(unique_groups, color_palette)}
            
            # Map sample indices to colors
            leaf_colors = {}
            for i, sample in enumerate(sample_names):
                group = sample_to_group.get(sample, 'Ungrouped')
                color = group_to_color[group]
                leaf_colors[i] = plt.matplotlib.colors.rgb2hex(color)
            
            # Create dendrogram with colored leaves
            dend = dendrogram(linkage_matrix, labels=sample_names, ax=ax, 
                            leaf_font_size=10, leaf_rotation=90)
            
            # Color the labels
            xlbls = ax.get_xmajorticklabels()
            for lbl in xlbls:
                sample_name = lbl.get_text()
                if sample_name in sample_to_group:
                    group = sample_to_group[sample_name]
                    lbl.set_color(plt.matplotlib.colors.rgb2hex(group_to_color[group]))
        else:
            dend = dendrogram(linkage_matrix, labels=sample_names, ax=ax,
                            leaf_font_size=10, leaf_rotation=90)
        
        ax.set_title(title)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Distance')
        
        plt.tight_layout()
        
        if save_name and self.figure_dir:
            filepath = self.figure_dir / save_name
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
        
        return fig
    
    def plot_metadata_correlation(self, correlation_results: Dict[str, Dict[str, float]],
                                 alpha: float = 0.05,
                                 title: str = "Metadata-Sequence Correlation Analysis",
                                 save_name: Optional[str] = None) -> plt.Figure:
        """Plot metadata correlation analysis results.
        
        Args:
            correlation_results: Results from MetadataAnalyzer
            alpha: Significance threshold
            title: Plot title
            save_name: Filename to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Prepare data
        variables = []
        statistics = []
        p_values = []
        var_types = []
        
        for var_name, results in correlation_results.items():
            variables.append(var_name)
            statistics.append(results['statistic'])
            p_values.append(results['p_value'])
            var_types.append(results['type'])
        
        # Convert to DataFrame
        plot_df = pd.DataFrame({
            'Variable': variables,
            'Statistic': statistics,
            'P-value': p_values,
            'Type': var_types
        })
        
        # Sort by p-value
        plot_df = plot_df.sort_values('P-value')
        
        # Plot 1: Bar plot of statistics
        colors = ['red' if p <= alpha else 'gray' for p in plot_df['P-value']]
        bars = ax1.barh(plot_df['Variable'], plot_df['Statistic'], color=colors)
        
        ax1.set_xlabel('Correlation Statistic')
        ax1.set_title('Correlation Strength')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add significance markers
        for i, (stat, p_val) in enumerate(zip(plot_df['Statistic'], plot_df['P-value'])):
            if p_val <= alpha:
                ax1.text(stat + 0.01, i, f'p={p_val:.3f}', va='center', fontsize=8)
        
        # Plot 2: -log10(p-value) plot
        neg_log_p = -np.log10(plot_df['P-value'])
        bars2 = ax2.barh(plot_df['Variable'], neg_log_p, color=colors)
        
        # Add significance threshold line
        threshold_line = -np.log10(alpha)
        ax2.axvline(x=threshold_line, color='red', linestyle='--', 
                   label=f'α = {alpha}')
        
        ax2.set_xlabel('-log10(P-value)')
        ax2.set_title('Statistical Significance')
        ax2.legend()
        
        fig.suptitle(title)
        plt.tight_layout()
        
        if save_name and self.figure_dir:
            filepath = self.figure_dir / save_name
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
        
        return fig
    
    def plot_grouping_evaluation(self, evaluation_results: Dict[str, Dict[str, float]],
                               title: str = "Grouping Strategy Evaluation",
                               save_name: Optional[str] = None) -> plt.Figure:
        """Plot grouping evaluation metrics.
        
        Args:
            evaluation_results: Dictionary of evaluation metrics for different groupings
            title: Plot title
            save_name: Filename to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Prepare data
        groupings = list(evaluation_results.keys())
        silhouette_scores = []
        within_distances = []
        between_distances = []
        n_groups = []
        
        for grouping, metrics in evaluation_results.items():
            silhouette_scores.append(metrics.get('silhouette_score', np.nan))
            within_distances.append(metrics.get('within_group_distance', np.nan))
            between_distances.append(metrics.get('between_group_distance', np.nan))
            n_groups.append(metrics.get('n_groups', 0))
        
        # Plot 1: Silhouette scores
        ax = axes[0, 0]
        bars = ax.bar(range(len(groupings)), silhouette_scores)
        ax.set_xticks(range(len(groupings)))
        ax.set_xticklabels(groupings, rotation=45, ha='right')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Clustering Quality')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Color bars based on score
        for bar, score in zip(bars, silhouette_scores):
            if not np.isnan(score):
                if score > 0.5:
                    bar.set_color('green')
                elif score > 0.25:
                    bar.set_color('yellow')
                else:
                    bar.set_color('red')
        
        # Plot 2: Within vs Between distances
        ax = axes[0, 1]
        x = np.arange(len(groupings))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, within_distances, width, label='Within-group')
        bars2 = ax.bar(x + width/2, between_distances, width, label='Between-group')
        
        ax.set_xticks(x)
        ax.set_xticklabels(groupings, rotation=45, ha='right')
        ax.set_ylabel('Average Distance')
        ax.set_title('Distance Comparison')
        ax.legend()
        
        # Plot 3: Separation ratio
        ax = axes[1, 0]
        separation_ratios = []
        for w, b in zip(within_distances, between_distances):
            if not np.isnan(w) and not np.isnan(b) and w > 0:
                separation_ratios.append(b / w)
            else:
                separation_ratios.append(np.nan)
        
        bars = ax.bar(range(len(groupings)), separation_ratios)
        ax.set_xticks(range(len(groupings)))
        ax.set_xticklabels(groupings, rotation=45, ha='right')
        ax.set_ylabel('Between/Within Distance Ratio')
        ax.set_title('Group Separation')
        ax.axhline(y=1, color='red', linestyle='--', label='No separation')
        ax.legend()
        
        # Plot 4: Number of groups
        ax = axes[1, 1]
        ax.bar(range(len(groupings)), n_groups)
        ax.set_xticks(range(len(groupings)))
        ax.set_xticklabels(groupings, rotation=45, ha='right')
        ax.set_ylabel('Number of Groups')
        ax.set_title('Group Count')
        
        fig.suptitle(title)
        plt.tight_layout()
        
        if save_name and self.figure_dir:
            filepath = self.figure_dir / save_name
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")
        
        return fig
    
    def generate_summary_report(self, distance_matrix: np.ndarray, sample_names: List[str],
                              correlation_results: Dict[str, Dict[str, float]],
                              grouping_evaluations: Dict[str, Dict[str, float]],
                              recommendation: Dict[str, Union[str, float, Dict]],
                              output_path: Union[str, Path]) -> None:
        """Generate a comprehensive HTML report with all analyses.
        
        Args:
            distance_matrix: Sample distance matrix
            sample_names: List of sample names
            correlation_results: Metadata correlation analysis results
            grouping_evaluations: Grouping evaluation results
            recommendation: Assembly strategy recommendation
            output_path: Path to save HTML report
        """
        output_path = Path(output_path)
        
        # Create temporary directory for figures
        temp_fig_dir = output_path.parent / 'report_figures'
        temp_fig_dir.mkdir(exist_ok=True)
        
        # Generate all figures
        self.figure_dir = temp_fig_dir
        
        # Distance heatmap
        self.plot_distance_heatmap(distance_matrix, sample_names, 
                                 save_name='distance_heatmap.png')
        
        # MDS plot
        self.plot_ordination(distance_matrix, sample_names, method='MDS',
                           save_name='mds_plot.png')
        
        # Dendrogram
        self.plot_dendrogram(distance_matrix, sample_names,
                           save_name='dendrogram.png')
        
        # Metadata correlations
        if correlation_results:
            self.plot_metadata_correlation(correlation_results,
                                         save_name='metadata_correlations.png')
        
        # Grouping evaluation
        if grouping_evaluations:
            self.plot_grouping_evaluation(grouping_evaluations,
                                        save_name='grouping_evaluation.png')
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sample Grouping Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                .recommendation {{ 
                    background-color: #f0f0f0; 
                    padding: 20px; 
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .metric {{ margin: 10px 0; }}
                img {{ max-width: 800px; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Sample Grouping Analysis Report</h1>
            
            <div class="recommendation">
                <h2>Assembly Strategy Recommendation</h2>
                <p><strong>Recommended Strategy:</strong> {recommendation['strategy'].upper()}</p>
                <p><strong>Reason:</strong> {recommendation['reason']}</p>
                <p><strong>Confidence:</strong> {recommendation['confidence']:.1%}</p>
            </div>
            
            <h2>Sample Distance Analysis</h2>
            <p>Analysis of {len(sample_names)} samples based on k-mer distances.</p>
            <img src="report_figures/distance_heatmap.png" alt="Distance Heatmap">
            <img src="report_figures/mds_plot.png" alt="MDS Plot">
            <img src="report_figures/dendrogram.png" alt="Dendrogram">
        """
        
        if correlation_results:
            html_content += """
            <h2>Metadata Correlation Analysis</h2>
            <img src="report_figures/metadata_correlations.png" alt="Metadata Correlations">
            
            <h3>Significant Variables (p < 0.05)</h3>
            <table>
                <tr>
                    <th>Variable</th>
                    <th>Type</th>
                    <th>Method</th>
                    <th>Statistic</th>
                    <th>P-value</th>
                </tr>
            """
            
            for var_name, results in sorted(correlation_results.items(), 
                                          key=lambda x: x[1]['p_value']):
                if results['p_value'] < 0.05:
                    html_content += f"""
                <tr>
                    <td>{var_name}</td>
                    <td>{results['type']}</td>
                    <td>{results['method']}</td>
                    <td>{results['statistic']:.3f}</td>
                    <td>{results['p_value']:.4f}</td>
                </tr>
                    """
            
            html_content += "</table>"
        
        if grouping_evaluations:
            html_content += """
            <h2>Grouping Strategy Evaluation</h2>
            <img src="report_figures/grouping_evaluation.png" alt="Grouping Evaluation">
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated summary report at {output_path}")