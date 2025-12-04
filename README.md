# Hurricane Wind Radius Prediction using LSTM

This repository provides a deep learning framework for predicting hurricane wind radius (R34) using LSTM neural networks with SHAP-based interpretability analysis. The project evaluates different feature groups to understand the contribution of meteorological variables to wind radius prediction.

## 📁 Project Structure
```
.
├── modules/
│   ├── __init__.py              # Module exports
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── model.py                 # LSTM model definition
│   ├── trainer.py               # Training and hyperparameter search
│   ├── shap_analysis.py         # SHAP value computation
│   └── visualization.py         # Plotting functions
├── data/
│   └── df_34out.csv             # Hurricane dataset
├── outputs/                      # Model outputs and figures (generated)
├── run_all_experiments.py       # Main script: runs all 3 experiments
├── requirements.txt
└── README.md
```

## 📋 Feature Groups

### Group 1: ERA5 Variables (23 features)
| Category | Features |
|----------|----------|
| Location | LAT, LON |
| Wind Structure | uv_max, rmax, rmax_avg, rmax_avg_adj, rmax_1, rmax_2, rmax_3, rmax_4 |
| Upper/Lower Level Winds | u200_mean, u200_std, u850_mean, u850_std, v200_mean, v200_std, v850_mean, v850_std |
| Thermal Structure | warm_core_diff_200_850, warm_core_pct_200_850 |
| Wind Shear | shear_u, shear_v, shear |

### Group 2: Observation Variables (7 features)
| Category | Features |
|----------|----------|
| Location | LAT, LON |
| Storm Characteristics | DIST2LAND, USA_WIND, USA_SSHS, STORM_SPEED, STORM_DIR |

### Group 3: Combined (28 features)
All features from Group 1 + Group 2

## 🔧 Setup and Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Data Preparation
Place your hurricane dataset (`df_34out.csv`) in the `data/` folder.

### Run Experiments
```bash
# Run all 3 experiments
python run_all_experiments.py

# Run specific group(s)
python run_all_experiments.py --groups group1_era5
python run_all_experiments.py --groups group2_obs group3_combined

# Custom paths
python run_all_experiments.py --data_path /path/to/data.csv --output_dir /path/to/output/
```

### Google Colab Usage
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Run experiments
!python run_all_experiments.py \
    --data_path '/content/drive/MyDrive/Hurricane Data/df_34out.csv' \
    --output_dir '/content/drive/MyDrive/Hurricane Data/'
```

## 📊 Output Structure

After execution, results will be saved in `outputs/`:

| File | Description |
|------|-------------|
| `model_{exp_name}.pth` | Trained model weights |
| `scaler_{exp_name}.pkl` | Fitted StandardScaler for deployment |
| `predictions_{exp_name}.csv` | Test set predictions |
| `scatter_{exp_name}.png` | Observed vs Predicted scatter plot |
| `feature_importance_{exp_name}.csv` | SHAP feature importance |
| `shap_bar_{exp_name}.png` | SHAP bar plot |

## 📈 Key Features

- **Unified Script**: Single `run_all_experiments.py` runs all 3 feature groups
- **LSTM Architecture**: Sequence modeling for hurricane temporal evolution
- **Hyperparameter Search**: Grid search with 5-fold cross-validation
- **SHAP Interpretability**: Global feature importance analysis
- **Visualization Suite**: Scatter plots and bar charts
- **Reproducibility**: Global random seed for consistent results
- **Deployment Ready**: Saves both model weights and scaler

## 🔬 Model Configuration

### Hyperparameter Search Space
| Parameter | Search Space |
|-----------|--------------|
| Hidden Size | 64, 128, 256 |
| Num Layers | 2, 3, 4 |
| Dropout | 0.0, 0.1, 0.3, 0.5 |
| Batch Size | 8, 16 |

### Training Settings
| Parameter | Value |
|-----------|-------|
| Test Ratio | 0.1 |
| K-Fold Splits | 5 |
| Max Epochs | 400 |
| Early Stopping Patience | 30 |
| Random Seed | 42 |

## 🔄 Model Deployment

To use a trained model for prediction on new data:
```python
import torch
import joblib
from modules import LSTMRegressor

# Load scaler and model
scaler = joblib.load('outputs/scaler_group1_era5.pkl')
model = LSTMRegressor(input_size=23, hidden_size=256, num_layers=2, dropout=0.0)
model.load_state_dict(torch.load('outputs/model_group1_era5.pth'))
model.eval()

# Preprocess new data
X_new_scaled = scaler.transform(X_new)
```

## 📈 Applications

This toolkit supports hurricane researchers and forecasters in:

- Predicting hurricane wind radius evolution
- Understanding key meteorological drivers of wind structure
- Evaluating the contribution of different feature sets
- Interpreting model predictions through SHAP analysis