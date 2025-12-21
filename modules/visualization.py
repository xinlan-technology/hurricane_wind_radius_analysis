"""
Visualization Module
- Scatter plot for actual vs predicted values
- SHAP bar plot for global feature importance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

# Feature name to label mapping
FEATURE_LABELS = {
    # Observation features
    'lat': 'Lat',
    'lon': 'Lon',
    'dist2land': 'D2L',
    'usa_wind': 'MSW_O',
    'usa_sshs': 'SSHS',
    'storm_speed': 'Spd',
    'storm_dir': 'Dir',

    # ERA5 features
    'uv_max': 'MSW_E',
    'rmax': 'Rmax',
    'rmax_avg': 'Rmax_avg',
    'rmax_avg_adj': 'Rmax_adj',
    'rmax_1': 'Rmax_1',
    'rmax_2': 'Rmax_2',
    'rmax_3': 'Rmax_3',
    'rmax_4': 'Rmax_4',
    'u200_mean': 'U200',
    'u200_std': 'U200_sd',
    'u850_mean': 'U850',
    'u850_std': 'U850_sd',
    'v200_mean': 'V200',
    'v200_std': 'V200_sd',
    'v850_mean': 'V850',
    'v850_std': 'V850_sd',
    'warm_core_diff_200_850': 'WC_diff',
    'warm_core_pct_200_850': 'WC_pct',
    'shear_u': 'SHR_u',
    'shear_v': 'SHR_v',
    'shear': 'SHR'
}


def plot_scatter(
    predictions_df: pd.DataFrame,
    output_path: str,
    n_features: int = None,
    figsize: tuple = (6, 6),
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
    
    # Auto select color based on feature count
    if n_features is None:
        color = '#2E86AB' # blue - Combined features
    elif n_features <= 10:
        color = '#E63946'  # red - Observed features
    elif n_features <= 25:
        color = '#F4A261'  # orange - ERA5 features
    else:
        color = '#2A9D8F'  # teal - Combined features
    
    # Create figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(
        actual,
        predicted,
        c=color,
        s=25,
        alpha=0.7,
        edgecolors='none'
    )
    
    # 1:1 line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    padding = (max_val - min_val) * 0.05
    line_range = [min_val - padding, max_val + padding]

    ax.plot(line_range, line_range, 'k--', linewidth=2, alpha=0.8)
    
    # Axis limits
    ax.set_xlim(line_range)
    ax.set_ylim(line_range)
    
    # Labels
    ax.set_xlabel('Observed R34 [km]', fontsize=18, fontweight='bold')
    ax.set_ylabel('Predicted R34 [km]', fontsize=18, fontweight='bold')
    
    # Square aspect
    ax.set_aspect('equal', adjustable='box')
    
    # Style
    ax.tick_params(axis='both', which='major', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
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
    figsize: tuple = (6, 6),
    feature_labels: Dict = FEATURE_LABELS,
    dpi: int = 300,
    top_n: int = 4
) -> None:
    """
    Create bar plot for global feature importance.
    Shows top N features + average of remaining features as "Others".
    
    Args:
        shap_results: Dictionary from compute_shap_values_lstm
        output_path: Path to save figure
        figsize: Figure size (default matches scatter plot)
        feature_labels: Dict mapping feature names to short labels
        dpi: Resolution
        top_n: Number of top features to show (default 4)
    """
    importance_df = shap_results['importance_df'].copy()
    n_features = len(importance_df)
    
    # Auto select color based on feature count (same as scatter plot)
    if n_features <= 10:
        color = '#E63946'  # red - Observed features
    elif n_features <= 25:
        color = '#F4A261'  # orange - ERA5 features
    else:
        color = '#2A9D8F'  # teal - Combined features
    
    # Convert feature names to lowercase
    importance_df['feature'] = importance_df['feature'].str.lower()
    
    # Apply short labels if provided
    if feature_labels:
        importance_df['feature'] = importance_df['feature'].map(
            lambda x: feature_labels.get(x, x)
        )
    
    # Get top N features and calculate "Others" (average of remaining)
    if len(importance_df) > top_n:
        top_df = importance_df.head(top_n).copy()
        others_avg = importance_df.iloc[top_n:]['importance'].mean()
        others_row = pd.DataFrame({'feature': ['Others'], 'importance': [others_avg]})
        plot_df = pd.concat([top_df, others_row], ignore_index=True)
    else:
        plot_df = importance_df.copy()
    
    # Keep order: most important first (rank 1, 2, 3, 4, Others)
    features = plot_df['feature'].values
    importance = plot_df['importance'].values
    n_bars = len(features)
    
    # Color scheme: main color for features, gray for "Others"
    colors = [color] * n_bars
    if 'Others' in features:
        others_idx = list(features).index('Others')
        colors[others_idx] = '#888888'
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set consistent margins for all plots
    fig.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.15)
    
    # Vertical bar plot
    bars = ax.bar(features, importance, color=colors, edgecolor='navy',
                  linewidth=0.5, width=0.6)
    
    # Value labels on top of bars
    for bar, val in zip(bars, importance):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + importance.max() * 0.02,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    
    # Labels - match scatter plot style (fontsize=18, bold)
    ax.set_xlabel('Feature', fontsize=18, fontweight='bold')
    ax.set_ylabel('Mean |SHAP Value|', fontsize=18, fontweight='bold')
    
    # Tick params - match scatter plot style (labelsize=14, bold)
    ax.tick_params(axis='both', which='major', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # Format y-axis to have consistent width (always show 2 decimal places)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    
    # Bold y-axis offset text (scientific notation)
    ax.yaxis.get_offset_text().set_fontsize(14)
    ax.yaxis.get_offset_text().set_fontweight('bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, importance.max() * 1.15)
    
    plt.savefig(output_path, dpi=dpi, facecolor='white')
    plt.close()
    
    print(f"[Plot] Saved SHAP bar plot to: {output_path}")
