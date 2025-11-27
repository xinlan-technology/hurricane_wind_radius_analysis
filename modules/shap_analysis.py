"""
SHAP Analysis Module
- compute_shap_values_lstm: Compute SHAP values for LSTM model
- aggregate_shap_to_global: Aggregate 3D SHAP values to global feature importance
- get_temporal_importance: Get feature importance at each time step
"""

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from typing import List, Dict
import random


def compute_shap_values_lstm(
    model: nn.Module,
    X_train_padded: np.ndarray,
    X_test_padded: np.ndarray,
    mask_test: np.ndarray,
    lengths_test: List[int],
    feature_names: List[str],
    device: torch.device,
    n_background: int = 100
) -> Dict:
    """
    Compute SHAP values for LSTM model using GradientExplainer.
    
    Args:
        model: Trained LSTMRegressor model
        X_train_padded: Padded training data (n_train, max_len, n_features)
        X_test_padded: Padded test data (n_test, max_len, n_features)
        mask_test: Mask for test data (n_test, max_len)
        lengths_test: Original lengths of test sequences
        feature_names: List of feature names
        device: torch device
        n_background: Number of background samples
        
    Returns:
        Dictionary with shap_values, mask, importance_df, temporal_importance, etc.
    """
    print("[SHAP] Computing SHAP values for LSTM model...")
    
    model.eval()
    model.to(device)
    
    # Select background samples
    np.random.seed(42)
    n_bg = min(n_background, len(X_train_padded))
    bg_indices = np.random.choice(len(X_train_padded), n_bg, replace=False)
    background = X_train_padded[bg_indices]
    
    print(f"[SHAP] Using {n_bg} background samples")
    print(f"[SHAP] Computing SHAP values for {len(X_test_padded)} test samples...")
    
    # Prepare tensors
    background_tensor = torch.tensor(background, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_padded, dtype=torch.float32).to(device)
    
    # GradientExplainer
    explainer = shap.GradientExplainer(model.forward_for_shap, background_tensor)
    shap_values = explainer.shap_values(X_test_tensor)
    
    # Handle output format
    # GradientExplainer returns list of length=time when output is (batch, time)
    # Each element is (n_samples, n_features)
    # Stack to (n_samples, max_len, n_features)
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=1)
        print(f"[SHAP] Stacked {shap_values.shape[1]} time steps")
    elif torch.is_tensor(shap_values):
        shap_values = shap_values.cpu().numpy()
    
    print(f"[SHAP] SHAP values shape: {shap_values.shape}")
    
    # Validate shape
    expected_shape = (len(X_test_padded), X_test_padded.shape[1], len(feature_names))
    if shap_values.shape != expected_shape:
        raise ValueError(f"Shape mismatch! Expected {expected_shape}, got {shap_values.shape}")
    
    # Compute global importance
    importance_df = aggregate_shap_to_global(shap_values, mask_test, feature_names)
    
    # Compute temporal importance
    temporal_importance = get_temporal_importance(shap_values, mask_test, feature_names)
    
    return {
        'shap_values': shap_values,
        'X_test': X_test_padded,
        'mask': mask_test,
        'lengths': lengths_test,
        'feature_names': feature_names,
        'importance_df': importance_df,
        'temporal_importance': temporal_importance
    }


def aggregate_shap_to_global(
    shap_values: np.ndarray,
    mask: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Aggregate 3D SHAP values to global feature importance.
    
    Args:
        shap_values: (n_samples, max_len, n_features)
        mask: (n_samples, max_len), 1=valid, 0=padding
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance ranking
    """
    # Apply mask to exclude padding positions
    mask_expanded = mask[:, :, np.newaxis]
    shap_masked = shap_values * mask_expanded
    
    # Mean absolute SHAP value per feature
    abs_shap_sum = np.abs(shap_masked).sum(axis=(0, 1))
    total_valid = mask.sum()
    mean_abs_shap = abs_shap_sum / total_valid
    
    # Create ranked DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    print("\n[SHAP] Global Feature Importance:")
    for _, row in importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return importance_df


def get_temporal_importance(
    shap_values: np.ndarray,
    mask: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Get feature importance at each time step.
    
    Args:
        shap_values: (n_samples, max_len, n_features)
        mask: (n_samples, max_len)
        feature_names: List of feature names
        
    Returns:
        DataFrame with shape (max_len, n_features)
    """
    n_samples, max_len, n_features = shap_values.shape
    
    temporal_importance = np.zeros((max_len, n_features))
    
    for t in range(max_len):
        valid_samples = mask[:, t] == 1
        if valid_samples.sum() > 0:
            temporal_importance[t, :] = np.abs(shap_values[valid_samples, t, :]).mean(axis=0)
    
    df = pd.DataFrame(temporal_importance, columns=feature_names)
    df.index.name = 'time_step'
    
    return df