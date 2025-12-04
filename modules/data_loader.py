"""
Data Loading Module
- Load CSV and filter hurricanes with >= min_observations
- Build X_sequences, y_sequences, hurricane_sids
- Create random trainval/test split
- Scale sequences
- Pad sequences for SHAP analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict


def load_data(
    file_path: str,
    feature_cols: List[str],
    target_col: str,
    min_observations: int = 3,
    time_col: str = 'TIME'
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], pd.DataFrame]:
    """
    Load hurricane data and build sequences.
    
    Args:
        file_path: Path to CSV file
        feature_cols: List of feature column names
        target_col: Name of target column (e.g., 'R34avg')
        min_observations: Minimum observations per hurricane
        time_col: Name of time column (default 'TIME')
    
    Returns:
        X_sequences: List of feature arrays
        y_sequences: List of target arrays  
        hurricane_sids: List of hurricane IDs
        df_seq: Processed DataFrame (needed for time values in evaluation)
    """
    df = pd.read_csv(file_path, low_memory=False)
    print(f"[Data] Loaded data shape: {df.shape}")
    
    # Fill missing target values with 0
    if df[target_col].isna().sum() > 0:
        print(f"[Data] Filling {df[target_col].isna().sum()} missing {target_col} values with 0")
        df[target_col] = df[target_col].fillna(0)
    
    # Sort by SID and time
    df_seq = df.sort_values(['SID', time_col]).copy()
    
    # Filter hurricanes with >= min_observations
    sid_counts = df_seq.groupby('SID').size()
    valid_sids = sid_counts[sid_counts >= min_observations].index
    df_seq = df_seq[df_seq['SID'].isin(valid_sids)].copy()
    
    print(f"[Data] Original hurricanes: {df['SID'].nunique()}")
    print(f"[Data] Filtered hurricanes (>= {min_observations} obs): {df_seq['SID'].nunique()}")
    print(f"[Data] Total observations: {len(df_seq)}")
    
    # Build sequences
    X_sequences = []
    y_sequences = []
    hurricane_sids = []
    
    for sid, group in df_seq.groupby('SID'):
        X_seq = group[feature_cols].values
        y_seq = group[target_col].values
        
        X_sequences.append(X_seq)
        y_sequences.append(y_seq)
        hurricane_sids.append(sid)
    
    print(f"[Data] Total sequences (hurricanes): {len(X_sequences)}")
    
    return X_sequences, y_sequences, hurricane_sids, df_seq


def create_random_split(
    X_sequences: List[np.ndarray],
    y_sequences: List[np.ndarray],
    hurricane_sids: List[str],
    test_ratio: float = 0.1,
    random_state: int = 42
) -> Dict:
    """
    Create random trainval/test split.
    Returns:
        Dictionary with trainval_indices, test_indices
    """
    n_total = len(X_sequences)
    all_indices = list(range(n_total))
    
    trainval_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_ratio,
        random_state=random_state
    )
    
    print(f"[Split] Random split: {len(trainval_indices)} trainval, {len(test_indices)} test")
    
    return {
        'trainval_indices': trainval_indices,
        'test_indices': test_indices
    }


def scale_sequences(
    X_sequences: List[np.ndarray],
    fit_indices: List[int]
) -> Tuple[List[np.ndarray], StandardScaler]:
    """
    Fit scaler on specified indices, transform ALL sequences.
    
    Args:
        X_sequences: List of feature arrays (unscaled)
        fit_indices: Indices used to fit the scaler
    
    Returns:
        X_sequences_scaled: All sequences scaled
        scaler: Fitted StandardScaler
    """
    # Fit scaler only on specified indices
    fit_features = np.vstack([X_sequences[i] for i in fit_indices])
    print(f"[Scale] Fitting scaler on {len(fit_indices)} sequences, shape: {fit_features.shape}")
    
    scaler = StandardScaler()
    scaler.fit(fit_features)
    
    # Transform all sequences
    X_sequences_scaled = []
    for x_seq in X_sequences:
        X_sequences_scaled.append(scaler.transform(x_seq))
    
    return X_sequences_scaled, scaler


def pad_sequences_for_shap(
    X_sequences: List[np.ndarray],
    indices: List[int] = None,
    max_len: int = None
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Pad variable-length sequences to fixed length for LSTM SHAP analysis.
    
    Args:
        X_sequences: List of feature arrays
        indices: Optional subset of indices to pad (default: all)
    
    Returns:
        X_padded: (n_samples, max_len, n_features)
        mask: (n_samples, max_len) - 1 for valid, 0 for padding
        lengths: List of original sequence lengths
    """
    if indices is None:
        indices = list(range(len(X_sequences)))
    
    sequences = [X_sequences[i] for i in indices]
    lengths = [len(seq) for seq in sequences]
    if max_len is None:
        max_len = max(lengths)
    n_features = sequences[0].shape[1]
    
    # Create padded array
    X_padded = np.zeros((len(sequences), max_len, n_features))
    mask = np.zeros((len(sequences), max_len))
    
    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_len) 
        X_padded[i, :seq_len, :] = seq[:seq_len]
        mask[i, :seq_len] = 1
    
    print(f"[SHAP] Padded to shape: {X_padded.shape}, max_len={max_len}")
    
    return X_padded, mask, lengths
