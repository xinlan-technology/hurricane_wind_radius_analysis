"""
Hurricane Wind Radius Prediction - Complete Pipeline
Runs all 3 experiments: Training + Evaluation + SHAP Analysis

Usage:
    python run_all_experiments.py
    python run_all_experiments.py --groups group1_era5 group2_obs
    python run_all_experiments.py --data_path /path/to/data.csv --output_dir /path/to/output/
"""

import os
import random
import numpy as np
import torch
import joblib

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
    plot_shap_bar
)

# Feature groups
GROUPS = {
    'group1_era5': [
        'LAT', 'LON',
        'uv_max', 'rmax', 'rmax_avg', 'rmax_avg_adj',
        'rmax_1', 'rmax_2', 'rmax_3', 'rmax_4',
        'u200_mean', 'u200_std', 'u850_mean', 'u850_std',
        'v200_mean', 'v200_std', 'v850_mean', 'v850_std',
        'warm_core_diff_200_850', 'warm_core_pct_200_850',
        'shear_u', 'shear_v', 'shear'
    ],
    'group2_obs': [
        'LAT', 'LON',
        'DIST2LAND', 'USA_WIND', 'USA_SSHS',
        'STORM_SPEED', 'STORM_DIR'
    ],
    'group3_combined': [
        'LAT', 'LON',
        'DIST2LAND', 'USA_WIND', 'USA_SSHS',
        'STORM_SPEED', 'STORM_DIR',
        'uv_max', 'rmax', 'rmax_avg', 'rmax_avg_adj',
        'rmax_1', 'rmax_2', 'rmax_3', 'rmax_4',
        'u200_mean', 'u200_std', 'u850_mean', 'u850_std',
        'v200_mean', 'v200_std', 'v850_mean', 'v850_std',
        'warm_core_diff_200_850', 'warm_core_pct_200_850',
        'shear_u', 'shear_v', 'shear'
    ]
}


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(exp_name, feature_cols, device, data_path, output_dir):
    """Run complete pipeline for a single experiment."""
    
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {exp_name}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print("=" * 60)
    
    # Step 1: Load Data
    print("\n[Step 1] Loading data...")
    X_sequences, y_sequences, hurricane_sids, df_seq = load_data(
        file_path=data_path,
        feature_cols=feature_cols,
        target_col='R34avg'
    )
    
    # Step 2: Create Random Split
    print("\n[Step 2] Creating random split...")
    split_info = create_random_split(X_sequences, y_sequences, hurricane_sids)
    trainval_indices = split_info['trainval_indices']
    test_indices = split_info['test_indices']
    
    # Step 3: Scale Sequences
    print("\n[Step 3] Scaling sequences...")
    X_sequences_scaled, scaler = scale_sequences(X_sequences, trainval_indices)
    
    # Save scaler for future deployment
    scaler_path = os.path.join(output_dir, f"scaler_{exp_name}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[Save] Scaler saved to: {scaler_path}")
    
    # Step 4: Hyperparameter Search
    print("\n[Step 4] Hyperparameter search with K-Fold CV...")
    best_params = hyperparameter_search(
        X_sequences=X_sequences,
        y_sequences=y_sequences,
        trainval_indices=trainval_indices,
        input_size=len(feature_cols),
        device=device
    )
    
    # Step 5: Train Final Model
    print("\n[Step 5] Training final model...")
    model_path = os.path.join(output_dir, f"model_{exp_name}.pth")
    model = train_final_model(
        X_sequences_scaled=X_sequences_scaled,
        y_sequences=y_sequences,
        trainval_indices=trainval_indices,
        best_params=best_params,
        input_size=len(feature_cols),
        device=device,
        model_save_path=model_path
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
        batch_size=best_params['batch_size']
    )
    predictions_df.to_csv(os.path.join(output_dir, f"predictions_{exp_name}.csv"), index=False)
    
    # Step 7: Scatter Plot
    print("\n[Step 7] Creating scatter plot...")
    metrics = plot_scatter(
        predictions_df=predictions_df,
        output_path=os.path.join(output_dir, f"scatter_{exp_name}.png"),
        n_features=len(feature_cols)
    )
    
    # Step 8: SHAP Analysis
    print("\n[Step 8] Running SHAP analysis...")
    train_lengths = [len(X_sequences_scaled[i]) for i in trainval_indices]
    test_lengths = [len(X_sequences_scaled[i]) for i in test_indices]
    max_len = max(max(train_lengths), max(test_lengths))
    
    X_train_padded, _, _ = pad_sequences_for_shap(X_sequences_scaled, trainval_indices, max_len)
    X_test_padded, mask_test, lengths_test = pad_sequences_for_shap(X_sequences_scaled, test_indices, max_len)
    
    shap_results = compute_shap_values_lstm(
        model=model,
        X_train_padded=X_train_padded,
        X_test_padded=X_test_padded,
        mask_test=mask_test,
        lengths_test=lengths_test,
        feature_names=feature_cols,
        device=device
    )
    
    shap_results['importance_df'].to_csv(os.path.join(output_dir, f"feature_importance_{exp_name}.csv"), index=False)
    plot_shap_bar(shap_results, os.path.join(output_dir, f"shap_bar_{exp_name}.png"))
    
    # Summary
    print("\n" + "-" * 40)
    print(f"{exp_name} COMPLETE")
    print(f"  R2: {metrics['r2']:.3f}, RMSE: {metrics['rmse']:.2f} km, MAE: {metrics['mae']:.2f} km")
    print(f"  Best params: {best_params}")
    print("-" * 40)
    
    return {'metrics': metrics, 'best_params': best_params}


def main(groups_to_run=None, data_path='./data/df_34out.csv', output_dir='./outputs/'):
    """Main function to run all experiments."""
    
    # Set global random seed for reproducibility
    set_seed(42)
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if groups_to_run is None:
        groups_to_run = list(GROUPS.keys())
    
    all_results = {}
    for exp_name in groups_to_run:
        if exp_name not in GROUPS:
            print(f"Warning: Unknown group '{exp_name}', skipping...")
            continue
        results = run_experiment(exp_name, GROUPS[exp_name], device, data_path, output_dir)
        all_results[exp_name] = results
    
    # Final Summary
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 60)
    for exp_name, results in all_results.items():
        m = results['metrics']
        p = results['best_params']
        print(f"\n{exp_name}: R2={m['r2']:.3f}, RMSE={m['rmse']:.2f} km, MAE={m['mae']:.2f} km")
        print(f"  Best: hidden={p['hidden_size']}, layers={p['num_layers']}, dropout={p['dropout']}, batch={p['batch_size']}")
    
    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', nargs='+', default=None)
    parser.add_argument('--data_path', type=str, default='./data/df_34out.csv')
    parser.add_argument('--output_dir', type=str, default='./outputs/')
    args = parser.parse_args()
    
    main(args.groups, args.data_path, args.output_dir)
