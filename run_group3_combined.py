"""
Hurricane Wind Radius Prediction - Group 3: Combined Variables
"""

import os
import torch
import numpy as np

from modules import (
    load_data,
    create_random_split,
    scale_sequences,
    pad_sequences_for_shap,
    hyperparameter_search,
    train_final_model,
    evaluate_on_test,
    plot_scatter,
    compute_shap_values_lstm,
    plot_shap_bar,
    plot_shap_heatmap
)

# Configuration
CONFIG = {
    'experiment_name': 'group3_combined',
    
    'feature_cols': [
        'LAT', 'LON',
        # Group 2 Basic variables
        'DIST2LAND', 'USA_WIND', 'USA_SSHS',
        'STORM_SPEED', 'STORM_DIR',
        # Group 1 ERA5 variables
        'uv_max', 'rmax', 'rmax_avg', 'rmax_avg_adj',
        'rmax_1', 'rmax_2', 'rmax_3', 'rmax_4',
        'u200_mean', 'u200_std', 'u850_mean', 'u850_std',
        'v200_mean', 'v200_std', 'v850_mean', 'v850_std',
        'warm_core_diff_200_850', 'warm_core_pct_200_850',
        'shear_u', 'shear_v', 'shear'
    ],
    
    'target_col': 'R34avg',
    'time_col': 'TIME',
    
    'data_path': './data/df_34out.csv',
    'output_dir': './outputs/',
    
    'test_ratio': 0.1,
    'random_state': 42,
    
    'param_grid': {
        'hidden_size': [64, 128, 256],
        'num_layers': [2, 3, 4],
        'dropout': [0.0, 0.1, 0.3, 0.5],
        'batch_size': [8, 16]
    },
}

# Main function
def main():
    
    exp_name = CONFIG['experiment_name']
    feature_cols = CONFIG['feature_cols']
    output_dir = CONFIG['output_dir']
    
    print("=" * 60)
    print(f"EXPERIMENT: {exp_name}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Load Data
    print("\n[Step 1] Loading data...")
    X_sequences, y_sequences, hurricane_sids, df_seq = load_data(
        file_path=CONFIG['data_path'],
        feature_cols=feature_cols,
        target_col=CONFIG['target_col'],
        min_observations=3,
        time_col=CONFIG['time_col']
    )
    input_size = len(feature_cols)
    
    # Step 2: Create Random Split
    print("\n[Step 2] Creating random split...")
    split_info = create_random_split(
        X_sequences=X_sequences,
        y_sequences=y_sequences,
        hurricane_sids=hurricane_sids,
        test_ratio=CONFIG['test_ratio'],
        random_state=CONFIG['random_state']
    )
    trainval_indices = split_info['trainval_indices']
    test_indices = split_info['test_indices']
    
    # Step 3: Scale Sequences
    print("\n[Step 3] Scaling sequences...")
    X_sequences_scaled, scaler = scale_sequences(
        X_sequences=X_sequences,
        trainval_indices=trainval_indices
    )
    
    # Step 4: Hyperparameter Search
    print("\n[Step 4] Hyperparameter search with K-Fold CV...")
    best_params = hyperparameter_search(
        X_sequences_scaled=X_sequences_scaled,
        y_sequences=y_sequences,
        trainval_indices=trainval_indices,
        input_size=input_size,
        device=device,
        param_grid=CONFIG['param_grid'],
        n_splits=5,
        random_state=CONFIG['random_state']
    )
    
    # Step 5: Train Final Model
    print("\n[Step 5] Training final model...")
    model_path = os.path.join(output_dir, f"model_{exp_name}.pth")
    model = train_final_model(
        X_sequences_scaled=X_sequences_scaled,
        y_sequences=y_sequences,
        trainval_indices=trainval_indices,
        best_params=best_params,
        input_size=input_size,
        device=device,
        model_save_path=model_path,
        max_epochs=400,
        patience=30
    )
    
    # Step 6: Evaluate on Test Set
    print("\n[Step 6] Evaluating on test set...")
    predictions_df = evaluate_on_test(
        model=model,
        X_sequences_scaled=X_sequences_scaled,
        y_sequences=y_sequences,
        test_indices=test_indices,
        hurricane_sids=hurricane_sids,
        df_seq=df_seq,
        device=device,
        batch_size=best_params['batch_size'],
        time_col=CONFIG['time_col']
    )
    
    pred_path = os.path.join(output_dir, f"predictions_{exp_name}.csv")
    predictions_df.to_csv(pred_path, index=False)
    print(f"[Save] Predictions saved to: {pred_path}")
    
    # Step 7: Scatter Plot
    print("\n[Step 7] Creating scatter plot...")
    scatter_path = os.path.join(output_dir, f"scatter_{exp_name}.png")
    metrics = plot_scatter(
        predictions_df=predictions_df,
        output_path=scatter_path
    )
    
    # Step 8: SHAP Analysis (LSTM)
    print("\n[Step 8] Running SHAP analysis...")
    
    X_train_padded, mask_train, lengths_train = pad_sequences_for_shap(
        X_sequences_scaled, trainval_indices
    )
    X_test_padded, mask_test, lengths_test = pad_sequences_for_shap(
        X_sequences_scaled, test_indices
    )
    
    shap_results = compute_shap_values_lstm(
        model=model,
        X_train_padded=X_train_padded,
        X_test_padded=X_test_padded,
        mask_test=mask_test,
        lengths_test=lengths_test,
        feature_names=feature_cols,
        device=device,
        n_background=100
    )
    
    importance_path = os.path.join(output_dir, f"feature_importance_{exp_name}.csv")
    shap_results['importance_df'].to_csv(importance_path, index=False)
    print(f"[Save] Feature importance saved to: {importance_path}")
    
    shap_bar_path = os.path.join(output_dir, f"shap_bar_{exp_name}.png")
    plot_shap_bar(shap_results, shap_bar_path)
    
    shap_heatmap_path = os.path.join(output_dir, f"shap_heatmap_{exp_name}.png")
    plot_shap_heatmap(shap_results, shap_heatmap_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Experiment: {exp_name}")
    print(f"Test R2: {metrics['r2']:.3f}")
    print(f"Test RMSE: {metrics['rmse']:.2f} km")
    print(f"Test MAE: {metrics['mae']:.2f} km")
    print(f"\nBest hyperparameters:")
    print(f"  hidden_size: {best_params['hidden_size']}")
    print(f"  num_layers: {best_params['num_layers']}")
    print(f"  dropout: {best_params['dropout']}")
    print(f"  batch_size: {best_params['batch_size']}")
    
    return {'metrics': metrics, 'best_params': best_params}


if __name__ == '__main__':
    results = main()
