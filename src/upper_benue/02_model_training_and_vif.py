import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch


# -----------------------------------------------------------------------------
# 1. SETUP & PATH RESOLUTION
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

csv_path = DATA_DIR / "UpperBenue_Flood_8000_Samples.csv"
df = pd.read_csv(csv_path)

# Target column
y = df['flood']

# Drop non-predictive metadata/spatial coordinates
drop_cols = ['system:index', '.geo', 'flood']
feature_cols = [col for col in df.columns if col not in drop_cols]
X = df[feature_cols]

print(f"Loaded {len(df)} samples with {X.shape[1]} initial features.")

# -----------------------------------------------------------------------------
# 2. VARIANCE INFLATION FACTOR (VIF) FILTERING
# -----------------------------------------------------------------------------
def calculate_vif(df_features):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_features)
    vif_df = pd.DataFrame()
    vif_df['Feature'] = df_features.columns
    vif_df['VIF'] = [
        variance_inflation_factor(scaled_data, i)
        for i in range(scaled_data.shape[1])
    ]
    return vif_df.sort_values(by='VIF', ascending=False)

vif_threshold = 10.0
X_vif = X.copy()

while True:
    vif_df = calculate_vif(X_vif)
    max_vif = vif_df['VIF'].iloc[0]
    if max_vif > vif_threshold:
        dropped_feature = vif_df['Feature'].iloc[0]
        print(f"Dropping '{dropped_feature}' with high VIF: {max_vif:.2f}")
        X_vif = X_vif.drop(columns=[dropped_feature])
    else:
        break

# Export Supplementary Table S1 (VIF)
vif_df.to_csv(RESULTS_DIR / "vif.csv", index=False)
print("\nFinal VIF Table saved to 'results/vif.csv':")
print(vif_df)

X_selected = X_vif.copy()

# -----------------------------------------------------------------------------
# 3. CORRELATION MATRIX HEATMAP
# -----------------------------------------------------------------------------
plt.figure(figsize=(12, 10), dpi=300)
corr_matrix = X_selected.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Feature Correlation Matrix (VIF Filtered)", fontsize=14, pad=12)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "fig_correlation.png", dpi=300)
plt.close()
print("\nCorrelation matrix figure saved to 'results/fig_correlation.png'.")

# -----------------------------------------------------------------------------
# 4. MODEL BENCHMARKING (EXPLICIT K-FOLD LOOP)
# -----------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics_summary = []

# --- CatBoost ---
print("\nTraining CatBoost via 5-Fold Cross-Validation...")
cat_metrics = {'auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

for fold, (train_idx, val_idx) in enumerate(cv.split(X_selected, y)):
    X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    cat_model = CatBoostClassifier(
        iterations=800,
        depth=8,
        learning_rate=0.04,
        l2_leaf_reg=3,
        verbose=0,
        random_seed=42
    )
    cat_model.fit(X_train, y_train)

    probs = cat_model.predict_proba(X_val)[:, 1]
    preds = cat_model.predict(X_val)

    cat_metrics['auc'].append(auc(*roc_curve(y_val, probs)[:2]))
    cat_metrics['accuracy'].append(accuracy_score(y_val, preds))
    cat_metrics['precision'].append(precision_score(y_val, preds))
    cat_metrics['recall'].append(recall_score(y_val, preds))
    cat_metrics['f1'].append(f1_score(y_val, preds))

metrics_summary.append({
    'Model': 'CatBoost',
    'AUC': np.mean(cat_metrics['auc']),
    'Accuracy': np.mean(cat_metrics['accuracy']),
    'Precision': np.mean(cat_metrics['precision']),
    'Recall': np.mean(cat_metrics['recall']),
    'F1': np.mean(cat_metrics['f1'])
})

# --- XGBoost ---
print("\nTraining XGBoost via 5-Fold Cross-Validation...")
xgb_metrics = {'auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

for fold, (train_idx, val_idx) in enumerate(cv.split(X_selected, y)):
    X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)

    probs = xgb_model.predict_proba(X_val)[:, 1]
    preds = xgb_model.predict(X_val)

    xgb_metrics['auc'].append(auc(*roc_curve(y_val, probs)[:2]))
    xgb_metrics['accuracy'].append(accuracy_score(y_val, preds))
    xgb_metrics['precision'].append(precision_score(y_val, preds))
    xgb_metrics['recall'].append(recall_score(y_val, preds))
    xgb_metrics['f1'].append(f1_score(y_val, preds))

metrics_summary.append({
    'Model': 'XGBoost',
    'AUC': np.mean(xgb_metrics['auc']),
    'Accuracy': np.mean(xgb_metrics['accuracy']),
    'Precision': np.mean(xgb_metrics['precision']),
    'Recall': np.mean(xgb_metrics['recall']),
    'F1': np.mean(xgb_metrics['f1'])
})

# --- TabNet ---
print("\nTraining TabNet via 5-Fold Cross-Validation...")
tabnet_metrics = {'auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

for fold, (train_idx, val_idx) in enumerate(cv.split(X_selected, y)):
    X_train, X_val = X_selected.iloc[train_idx].values, X_selected.iloc[val_idx].values
    y_train, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    tabnet = TabNetClassifier(
        n_d=32, n_a=32, n_steps=5, gamma=1.5,
        cat_idxs=[], cat_dims=[], cat_emb_dim=1,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params=dict(step_size=10, gamma=0.9),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax", verbose=0
    )

    tabnet.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        eval_name=['val'],
        eval_metric=['auc'],
        max_epochs=120,
        patience=15,
        batch_size=256,
        virtual_batch_size=128
    )

    probs = tabnet.predict_proba(X_val_scaled)[:, 1]
    preds = tabnet.predict(X_val_scaled)

    tabnet_metrics['auc'].append(auc(*roc_curve(y_val, probs)[:2]))
    tabnet_metrics['accuracy'].append(accuracy_score(y_val, preds))
    tabnet_metrics['precision'].append(precision_score(y_val, preds))
    tabnet_metrics['recall'].append(recall_score(y_val, preds))
    tabnet_metrics['f1'].append(f1_score(y_val, preds))

metrics_summary.append({
    'Model': 'TabNet',
    'AUC': np.mean(tabnet_metrics['auc']),
    'Accuracy': np.mean(tabnet_metrics['accuracy']),
    'Precision': np.mean(tabnet_metrics['precision']),
    'Recall': np.mean(tabnet_metrics['recall']),
    'F1': np.mean(tabnet_metrics['f1'])
})

# Export Model Performance Metrics
df_metrics = pd.DataFrame(metrics_summary)
df_metrics.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
print("\nFinal Model Metrics saved to 'results/model_metrics.csv':")
print(df_metrics.round(4))

# -----------------------------------------------------------------------------
# 5. ROC CURVE PLOTTING
# -----------------------------------------------------------------------------
plt.figure(figsize=(8, 6), dpi=300)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42, stratify=y)

# 1. CatBoost & XGBoost
eval_models = {
    'CatBoost': CatBoostClassifier(iterations=800, depth=8, learning_rate=0.04, l2_leaf_reg=3, verbose=0, random_seed=42),
    'XGBoost': XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.05, random_state=42, eval_metric='logloss')
}

colors = {'CatBoost': '#1f77b4', 'XGBoost': '#2ca02c', 'TabNet': '#ff7f0e'}

for name, model in eval_models.items():
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, color=colors[name], label=f"{name} (AUC = {roc_auc:.3f})")

# 2. TabNet
print("\nFitting TabNet on 70/30 holdout test set for ROC curve...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)

tabnet = TabNetClassifier(
    n_d=32, n_a=32, n_steps=5, gamma=1.5,
    cat_idxs=[], cat_dims=[], cat_emb_dim=1,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params=dict(step_size=10, gamma=0.9),
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type="entmax", verbose=0,
    seed=42
)
tabnet.fit(
    X_train_scaled, y_train.values,
    eval_set=[(X_test_scaled, y_test.values)],
    eval_name=['test'],
    eval_metric=['auc'],
    max_epochs=120,
    patience=15,
    batch_size=256,
    virtual_batch_size=128
)
probs_tabnet = tabnet.predict_proba(X_test_scaled)[:, 1]
fpr_tn, tpr_tn, _ = roc_curve(y_test.values, probs_tabnet)
roc_auc_tn = auc(fpr_tn, tpr_tn)
plt.plot(fpr_tn, tpr_tn, lw=2, color=colors['TabNet'], label=f"TabNet (AUC = {roc_auc_tn:.3f})")

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier (AUC = 0.500)')
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.05)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, pad=10)
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "fig_roc.png", dpi=300)
plt.close()
print("\nROC curves figure saved to 'results/fig_roc.png'.")