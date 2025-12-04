"""
Trainer Module
- train_and_evaluate: K-Fold CV training
- hyperparameter_search: Grid search
- train_final_model: Final training on all trainval
- evaluate_on_test: Test evaluation with predictions export
"""

import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple

from .model import LSTMRegressor, HurricaneDataset, collate_fn, masked_mse_loss


def train_and_evaluate(model, train_loader, val_loader, optimizer, device, max_epochs=400, patience=20):
    """
    Training and evaluation function.
    Used for K-Fold cross-validation.
    """
    model.to(device)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Early stopping setup
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(max_epochs):

        # Training phase
        model.train()
        for X_batch, y_batch, mask_batch, lengths_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            lengths_batch = lengths_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch, lengths_batch)

            loss = masked_mse_loss(pred, y_batch, mask_batch)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation phase - global weighted average
        model.eval()
        total_squared_error = 0.0
        total_samples = 0.0

        with torch.no_grad():
            for X_batch, y_batch, mask_batch, lengths_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)
                lengths_batch = lengths_batch.to(device)

                pred = model(X_batch, lengths_batch)

                squared_error = ((pred - y_batch) ** 2 * mask_batch).sum().item()
                num_samples = mask_batch.sum().item()

                total_squared_error += squared_error
                total_samples += num_samples

        # Global average
        avg_val_loss = total_squared_error / max(total_samples, 1.0)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return best_val_loss


def hyperparameter_search(
    X_sequences: List[np.ndarray],
    y_sequences: List[np.ndarray],
    trainval_indices: List[int],
    input_size: int,
    device: torch.device,
    param_grid: Dict = None,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Hyperparameter search with K-Fold CV.
    
    Returns:
        best_params: Dictionary with best hyperparameters
    """
    if param_grid is None:
        param_grid = {
            'hidden_size': [64, 128, 256],
            'num_layers': [2, 3, 4],
            'dropout': [0.0, 0.1, 0.3, 0.5],
            'batch_size': [8, 16]
        }
    
    # Get all combinations
    param_combinations = list(itertools.product(
        param_grid['hidden_size'],
        param_grid['num_layers'],
        param_grid['dropout'],
        param_grid['batch_size']
    ))
    
    print(f"[Search] Total hyperparameter combinations: {len(param_combinations)}")
    
    # K-Fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    results = []
    
    for comb_idx, (hidden_size, num_layers, dropout, batch_size) in enumerate(param_combinations):
        fold_losses = []
        
        # Print progress
        if (comb_idx + 1) % 20 == 0:
            print(f"[Search] Progress: {comb_idx + 1}/{len(param_combinations)} combinations tested")
        
        # K-Fold cross-validation within trainval_indices
        for fold_idx, (train_rel, val_rel) in enumerate(kf.split(trainval_indices)):

            # Map indices back to original
            train_indices = [trainval_indices[i] for i in train_rel]
            val_indices = [trainval_indices[i] for i in val_rel]
            
            # Fit scaler only on this fold's training data (prevent data leakage)
            train_features = np.vstack([X_sequences[i] for i in train_indices])
            scaler = StandardScaler()
            scaler.fit(train_features)
            
            # Scale train and val separately using this fold's scaler
            X_train = [scaler.transform(X_sequences[i]) for i in train_indices]
            y_train = [y_sequences[i] for i in train_indices]
            X_val = [scaler.transform(X_sequences[i]) for i in val_indices]
            y_val = [y_sequences[i] for i in val_indices]
            
            train_dataset = HurricaneDataset(X_train, y_train)
            val_dataset = HurricaneDataset(X_val, y_val)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size,
                shuffle=True, collate_fn=collate_fn
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size,
                shuffle=False, collate_fn=collate_fn
            )
            
            # Create model
            model = LSTMRegressor(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            ).to(device)
            
            # Optimizer
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Train and evaluate
            val_loss = train_and_evaluate(
                model, train_loader, val_loader, optimizer, device,
                max_epochs=400, patience=20
            )
            
            fold_losses.append(val_loss)
        
        # Average loss across folds
        avg_loss = sum(fold_losses) / len(fold_losses)
        
        results.append({
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'batch_size': batch_size,
            'avg_val_loss': avg_loss
        })
    
    # Sort and find best
    sorted_results = sorted(results, key=lambda x: x['avg_val_loss'])
    best = sorted_results[0]
    
    print(f"\n[Search] Best hyperparameters:")
    print(f"  hidden_size={best['hidden_size']}")
    print(f"  num_layers={best['num_layers']}")
    print(f"  dropout={best['dropout']}")
    print(f"  batch_size={best['batch_size']}")
    print(f"  Best validation loss: {best['avg_val_loss']:.4f}")
    
    return best


def train_final_model(
    X_sequences_scaled: List[np.ndarray],
    y_sequences: List[np.ndarray],
    trainval_indices: List[int],
    best_params: Dict,
    input_size: int,
    device: torch.device,
    model_save_path: str,
    max_epochs: int = 400,
    patience: int = 30
) -> nn.Module:
    """
    Train final model with best hyperparameters.
    
    Returns:
        Trained model
    """
    print(f"\n[Train] Training final model...")
    
    # Create datasets for final training
    X_trainval = [X_sequences_scaled[i] for i in trainval_indices]
    y_trainval = [y_sequences[i] for i in trainval_indices]
    trainval_dataset = HurricaneDataset(X_trainval, y_trainval)
    
    # Create data loader
    trainval_loader = DataLoader(
        trainval_dataset,
        batch_size=best_params['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Create model
    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training
    best_train_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(max_epochs):
        model.train()
        
        total_squared_error = 0.0
        total_samples = 0.0
        
        for X_batch, y_batch, mask_batch, lengths_batch in trainval_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            lengths_batch = lengths_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(X_batch, lengths_batch)
            
            loss = masked_mse_loss(pred, y_batch, mask_batch)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate statistics
            with torch.no_grad():
                squared_error = ((pred - y_batch) ** 2 * mask_batch).sum().item()
                num_samples = mask_batch.sum().item()
                total_squared_error += squared_error
                total_samples += num_samples
        
        avg_train_loss = total_squared_error / max(total_samples, 1.0)
        
        if (epoch + 1) % 50 == 0:
            print(f"[Train] Epoch {epoch+1}/{max_epochs}, Loss: {avg_train_loss:.4f}")
        
        scheduler.step(avg_train_loss)
        
        # Early stopping
        if avg_train_loss < best_train_loss - 1e-6:
            best_train_loss = avg_train_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Train] Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(model_save_path))
    print(f"[Train] Best training loss: {best_train_loss:.4f}")
    
    return model


def evaluate_on_test(
    model: nn.Module,
    X_sequences_scaled: List[np.ndarray],
    y_sequences: List[np.ndarray],
    test_indices: List[int],
    hurricane_sids: List[str],
    df_seq: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    time_col: str = 'TIME'
) -> pd.DataFrame:
    """
    Evaluate on test set and return predictions DataFrame.
    
    Returns:
        DataFrame with SID, time, actual_R34avg, predicted_R34avg, error, squared_error
    """
    print(f"\n[Eval] Evaluating on test set...")
    
    # Create test dataset
    X_test = [X_sequences_scaled[i] for i in test_indices]
    y_test = [y_sequences[i] for i in test_indices]
    test_dataset = HurricaneDataset(X_test, y_test)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    model.eval()
    
    total_squared_error = 0.0
    total_samples = 0.0
    test_predictions_by_hurricane = []
    
    test_sids = [hurricane_sids[i] for i in test_indices]
    sample_ptr = 0
    
    with torch.no_grad():
        for X_batch, y_batch, mask_batch, lengths_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            lengths_batch = lengths_batch.to(device)
            
            pred = model(X_batch, lengths_batch)
            
            # Accumulate MSE
            squared_error = ((pred - y_batch) ** 2 * mask_batch).sum().item()
            num_samples = mask_batch.sum().item()
            total_squared_error += squared_error
            total_samples += num_samples
            
            # Process each sample
            B = X_batch.size(0)
            for b in range(B):
                sid = test_sids[sample_ptr]
                L = int(lengths_batch[b].item())
                
                # Get time values
                hurricane_data = df_seq[df_seq['SID'] == sid].sort_values(time_col)
                time_values = hurricane_data[time_col].values[:L]
                
                y_valid = y_batch[b, :L].detach().cpu().numpy()
                p_valid = pred[b, :L].detach().cpu().numpy()
                
                for t in range(L):
                    test_predictions_by_hurricane.append({
                        'SID': sid,
                        'time': time_values[t],
                        'actual_R34avg': float(y_valid[t]),
                        'predicted_R34avg': float(p_valid[t]),
                    })
                
                sample_ptr += 1
    
    # Calculate metrics
    avg_test_loss = total_squared_error / max(total_samples, 1.0)
    print(f"[Eval] Test MSE: {avg_test_loss:.4f}, RMSE: {np.sqrt(avg_test_loss):.4f}")
    
    # Create DataFrame
    predictions_df = pd.DataFrame(test_predictions_by_hurricane)
    predictions_df['error'] = predictions_df['predicted_R34avg'] - predictions_df['actual_R34avg']
    predictions_df['squared_error'] = predictions_df['error'] ** 2
    
    return predictions_df
