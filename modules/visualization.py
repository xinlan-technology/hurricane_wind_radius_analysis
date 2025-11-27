"""
Visualization Module
- Scatter plot for actual vs predicted values
- SHAP bar plot for global feature importance
- SHAP heatmap for temporal feature importance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict


def plot_scatter(
    predictions_df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (8, 8),
    dpi: int = 300
) -> Dict:
    """
    Create scatter plot for actual vs predicted values.
    
    Args:
        predictions_df: DataFrame with 'actual_R34avg' and 'predicted_R34avg' columns
        output_path: Path to save figure
        figsize: Figure size
        dpi: Resolution
        
    Returns:
        Dictionary with metrics (R2, RMSE, MAE, etc.)
    """
    actual = predictions_df['actual_R34avg'].values
    predicted = predictions_df['predicted_R34avg'].values
    
    # Calculate metrics
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    
    # Create figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate density for coloring
    xy = np.vstack([actual, predicted])
    try:
        density = stats.gaussian_kde(xy)(xy)
        idx = density.argsort()
        actual_sorted = actual[idx]
        predicted_sorted = predicted[idx]
        density_sorted = density[idx]
        
        scatter = ax.scatter(
            actual_sorted,
            predicted_sorted,
            c=density_sorted,
            cmap='viridis',
            s=25,
            alpha=0.7,
            edgecolors='none'
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Density', fontsize=12, fontweight='bold')
    except:
        ax.scatter(
            actual,
            predicted,
            c='#2E86AB',
            s=25,
            alpha=0.6,
            edgecolors='none'
        )
    
    # 1:1 line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    padding = (max_val - min_val) * 0.05
    line_range = [min_val - padding, max_val + padding]
    
    ax.plot(line_range, line_range, 'k--', linewidth=2, alpha=0.8, label='1:1 Line')
    
    # Axis limits
    ax.set_xlim(line_range)
    ax.set_ylim(line_range)
    
    # Labels
    ax.set_xlabel('Observed R34 (km)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted R34 (km)', fontsize=14, fontweight='bold')
    
    # Legend
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    # Square aspect
    ax.set_aspect('equal', adjustable='box')
    
    # Style
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[Plot] Saved scatter plot to: {output_path}")
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'n_samples': len(actual)
    }


def plot_shap_bar(
    shap_results: Dict,
    output_path: str,
    figsize: tuple = (10, 8),
    dpi: int = 300
) -> None:
    """
    Create bar plot for global feature importance.
    
    Args:
        shap_results: Dictionary from compute_shap_values_lstm
        output_path: Path to save figure
        figsize: Figure size
        dpi: Resolution
    """
    importance_df = shap_results['importance_df'].copy()
    
    # Convert feature names to lowercase
    importance_df['feature'] = importance_df['feature'].str.lower()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Reverse for bottom-to-top display
    features = importance_df['feature'].values[::-1]
    importance = importance_df['importance'].values[::-1]
    
    # Color gradient
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))[::-1]
    
    bars = ax.barh(features, importance, color=colors, edgecolor='navy',
                   linewidth=0.5, height=0.7)
    
    # Value labels
    for bar, val in zip(bars, importance):
        ax.text(bar.get_width() + importance.max() * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', ha='left',
                fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Mean |SHAP Value|', fontsize=14, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=14, fontweight='bold')
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, importance.max() * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[Plot] Saved SHAP bar plot to: {output_path}")


def plot_shap_heatmap(
    shap_results: Dict,
    output_path: str,
    figsize: tuple = (14, 8),
    dpi: int = 300,
    max_time_steps: int = None,
    min_valid_samples: int = 5,
    time_interval: int = 6
) -> None:
    """
    Create heatmap showing feature importance over time.
    
    Args:
        shap_results: Dictionary from compute_shap_values_lstm
        output_path: Path to save figure
        figsize: Figure size
        dpi: Resolution
        max_time_steps: Maximum number of time steps to show
        min_valid_samples: Minimum valid samples required for a time step
        time_interval: Hours between each time step (default 6)
    """
    temporal_df = shap_results['temporal_importance'].copy()
    
    # Convert column names (feature names) to lowercase
    temporal_df.columns = temporal_df.columns.str.lower()
    
    mask = shap_results['mask']
    
    # Count valid samples at each time step
    valid_counts = mask.sum(axis=0)
    
    # Filter time steps with enough valid samples
    valid_timesteps = valid_counts >= min_valid_samples
    temporal_df = temporal_df[valid_timesteps]
    
    # Limit time steps if specified
    if max_time_steps and len(temporal_df) > max_time_steps:
        temporal_df = temporal_df.iloc[:max_time_steps, :]
    
    # Remove all-zero rows
    non_zero = temporal_df.sum(axis=1) > 0
    temporal_df = temporal_df[non_zero]
    
    # Transpose: features as rows, time as columns
    heatmap_data = temporal_df.T
    
    # Rename columns to actual hours
    n_steps = len(heatmap_data.columns)
    heatmap_data.columns = [i * time_interval for i in range(1, n_steps + 1)]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap='YlOrRd',
        annot=False,
        linewidths=0.5,
        cbar_kws={'label': 'Mean |SHAP Value|'}
    )
    
    # Style colorbar label
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_fontsize(12)
    cbar.ax.yaxis.label.set_fontweight('bold')
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_xlabel('Time (hours)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=14, fontweight='bold')
    
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Rotate x-axis labels if many time steps
    if n_steps > 15:
        ax.set_xticks(ax.get_xticks()[::2])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[Plot] Saved SHAP heatmap to: {output_path}")