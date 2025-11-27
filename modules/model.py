"""
Model Module
- HurricaneDataset
- collate_fn
- LSTMRegressor
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from typing import List, Tuple


class HurricaneDataset(Dataset):
    """Custom Dataset class for PyTorch."""
    
    def __init__(self, X_sequences, y_sequences):
        self.X_sequences = [torch.tensor(x, dtype=torch.float32) for x in X_sequences]
        self.y_sequences = [torch.tensor(y, dtype=torch.float32) for y in y_sequences]

    def __len__(self):
        return len(self.X_sequences)

    def __getitem__(self, idx):
        return self.X_sequences[idx], self.y_sequences[idx]


def collate_fn(batch):
    """Collate function for padding variable-length sequences."""
    X_list, y_list = zip(*batch)

    # Pad sequences to same length within batch
    X_padded = pad_sequence(X_list, batch_first=True, padding_value=0.0)
    y_padded = pad_sequence(y_list, batch_first=True, padding_value=0.0)

    # Create mask for valid positions
    lengths = torch.tensor([len(x) for x in X_list], dtype=torch.long)
    mask = torch.arange(y_padded.size(1))[None, :] < lengths[:, None]
    mask = mask.float()

    return X_padded, y_padded, mask, lengths


class LSTMRegressor(nn.Module):
    """LSTM Model Class."""
    
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        # Pack sequence to ignore padding (for training)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)

        # Unpack sequence
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        return self.fc(out).squeeze(-1)
    
    def forward_for_shap(self, x):
        """
        Simplified forward for SHAP analysis (no lengths parameter).
        Assumes all timesteps are valid (no packing).
        
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, seq_len) predictions
        """
        out, _ = self.lstm(x)
        return self.fc(out).squeeze(-1)


def masked_mse_loss(pred, target, mask):
    """Calculate masked MSE loss."""
    squared_error = (pred - target) ** 2
    masked_error = squared_error * mask
    return masked_error.sum() / mask.sum().clamp(min=1)
