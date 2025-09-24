import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split

import torch
import torchtuples as tt

from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
from sklearn.metrics import roc_curve, auc
import sksurv.util
import joblib
import os

# Set random seeds
np.random.seed(1234)
torch.manual_seed(123)

# Load single LCRN dataset
data_path = '/Users/ryanlim/Desktop/University/IMGS/Cohort HCC Score/Datasets/LCRN Validation Data.xlsx'
df_lcrn = pd.read_excel(data_path)

# Define variables
features = [
    'age60', 'newdefinition', 'obese', 't2diabetes', 'listascites', 'HEwaitlist',
    'init_albumin', 'init_inr', 'init_bilirubin', 'init_serum_creat',
    'maxafp', 'maxtumornumber', 'maxtumorsize', 'sumtumoursize'
]
target_cols = ['time', 'status']

# Drop missing and clean
for df in [df_lcrn]:
    df.replace({',': '.'}, regex=True, inplace=True)  # convert commas to dots for decimals
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=features + target_cols, inplace=True)

# Log-transform AFP
for df in [df_lcrn]:
    df['logAFP'] = np.log1p(df['maxafp'])

features = [f for f in features if f != 'maxafp'] + ['logAFP']

# Split LCRN into train/val (80:20)
df_train, df_val = train_test_split(
    df_lcrn,
    test_size=0.3,
    random_state=1234,
    stratify=df_lcrn['status']
)

# Preprocessing
cols_standardize = ['init_albumin', 'init_inr', 'init_bilirubin', 'init_serum_creat',
                    'logAFP', 'maxtumorsize', 'sumtumoursize']
cols_leave = list(set(features) - set(cols_standardize))

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]
x_mapper = DataFrameMapper(standardize + leave)

# Transform datasets
x_train = x_mapper.fit_transform(df_train).astype('float32')
y_train = (df_train['time'].values, df_train['status'].values)

x_val = x_mapper.transform(df_val).astype('float32')
y_val = (df_val['time'].values, df_val['status'].values)

# durations and events for evaluation
durations_train, events_train = y_train
durations_val, events_val = y_val

# Define model
in_features = x_train.shape[1]
net = tt.practical.MLPVanilla(in_features, [64, 16], 1, batch_norm=True, dropout=0.1, output_bias=False)

class AdamWithWeightDecay(tt.optim.Adam):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('weight_decay', 0.005)
        super().__init__(*args, **kwargs)

model = CoxPH(net, AdamWithWeightDecay)

# Train
batch_size = 30
lrfinder = model.lr_finder(x_train, y_train, batch_size, tolerance=10)
print(f"Best Learning Rate: {lrfinder.get_best_lr():.4f}")
model.optimizer.set_lr(0.0001)
log = model.fit(x_train, y_train, batch_size, epochs=200,
          callbacks=[], verbose=True, 
          val_data=(x_val, y_val), val_batch_size=batch_size)

# Plot training and validation loss
log.plot()
plt.show()

# Predict survival
model.compute_baseline_hazards()
surv_train = model.predict_surv_df(x_train)
surv_val = model.predict_surv_df(x_val)

# Evaluate C-index
ev_train = EvalSurv(surv_train, durations_train, events_train, censor_surv='km')
c_index_train = ev_train.concordance_td()
print(f"Concordance Index (Train - LCRN): {c_index_train:.4f}")

ev_val = EvalSurv(surv_val, durations_val, events_val, censor_surv='km')
c_index_val = ev_val.concordance_td()
print(f"Concordance Index (Validation - LCRN): {c_index_val:.4f}")

# Evaluate Brier Score
time_grid_train = np.linspace(durations_train.min(), durations_train.max(), 100)
brier_score_train = ev_train.integrated_brier_score(time_grid_train)
print(f"Brier Score (Train - LCRN): {brier_score_train:.4f}")

time_grid_val = np.linspace(durations_val.min(), durations_val.max(), 100)
brier_score_val = ev_val.integrated_brier_score(time_grid_val)
print(f"Brier Score (Validation - LCRN): {brier_score_val:.4f}")


# -------------------------------------------------------------
# Helper: ROC & AUC at a single landmark time
# -------------------------------------------------------------
from sklearn.metrics import roc_curve, auc

def roc_at_time(surv_df, y_tuple, time_pt):
    """
    surv_df : DataFrame from model.predict_surv_df
    y_tuple : (durations, events)  –  same order as y_val / y_test
    time_pt : landmark time in same units as durations
    Returns  fpr, tpr, auc_val, used_time
    """
    times, events = y_tuple
    # nearest time in prediction grid
    idx_near = surv_df.index.get_loc(time_pt, method="nearest")
    used_time = surv_df.index[idx_near]

    # predicted risk = 1 - S(used_time)
    risk = 1.0 - surv_df.loc[used_time].values

    # binary label: event occurred by time_pt
    label = (times <= time_pt) & (events == 1)

    fpr, tpr, _ = roc_curve(label, risk)
    auc_val = auc(fpr, tpr)
    return fpr, tpr, auc_val, used_time

# -------------------------------------------------------------
# Plot ROC curves for 12, 36, 60, 120 months
# -------------------------------------------------------------
target_months = [12, 36, 60, 120]
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for ax, t in zip(axes.flatten(), target_months):
    # --- validation ---
    fpr_v, tpr_v, auc_v, used_v = roc_at_time(surv_val, y_val, t)
    # --- train ---
    fpr_t, tpr_t, auc_t, used_t = roc_at_time(surv_train, y_train, t)

    ax.plot(fpr_t, tpr_t, lw=2, label=f'Train AUC={auc_t:.3f}')
    ax.plot(fpr_v, tpr_v, lw=2, label=f'Validation AUC={auc_v:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title(f't = {used_v:.0f} months')  # shows actual nearest time
    ax.grid(False)
    ax.legend(loc='lower right')

plt.suptitle('ROC Curves at 12, 36, 60, 120 months', y=1.03, fontsize=14)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# Helper : calibration curve at a fixed landmark time (KM‑based)
# ------------------------------------------------------------------
from lifelines import KaplanMeierFitter

def calibration_by_km(surv_df, y_tuple, landmark, n_bins=10):
    """
    Return DataFrame with columns [bin, pred, obs] where
      pred = mean predicted risk  (1 - Ŝ(landmark))
      obs  = observed event rate  by KM at landmark.
    """
    times, events = y_tuple

    # find nearest prediction row to landmark
    nearest_idx = surv_df.index.get_loc(landmark, method="nearest")
    used_time = surv_df.index[nearest_idx]

    pred_risk = 1.0 - surv_df.loc[used_time].values  # risk scores

    df_tmp = pd.DataFrame({"pred": pred_risk,
                           "time": times,
                           "event": events})
    # bin by deciles of predicted risk
    df_tmp["bin"] = pd.qcut(df_tmp["pred"], n_bins,
                            labels=False, duplicates="drop")

    out = []
    kmf = KaplanMeierFitter()
    for b in sorted(df_tmp["bin"].unique()):
        sub = df_tmp[df_tmp["bin"] == b]
        kmf.fit(sub["time"], sub["event"])
        obs = 1.0 - kmf.predict(landmark)        # observed event P
        out.append({"bin": b,
                    "pred": sub["pred"].mean(),
                    "obs": obs})
    cal_df = pd.DataFrame(out)
    return cal_df, used_time

# ------------------------------------------------------------------
# ❷ Multi‑panel calibration curves at 12, 36, 60, 120 months
# ------------------------------------------------------------------
landmarks = [12.0, 36.0, 60.0, 120.0]          # months you care about
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for ax, lm in zip(axes.flatten(), landmarks):

    # Skip if landmark exceeds max follow‑up
    if lm > surv_val.index.max():
        ax.set_axis_off()
        ax.set_title(f"{lm:.0f}m (out of range)")
        continue

    # --- compute calibration for val & train
    cal_val, used_val = calibration_by_km(surv_val, y_val, lm, n_bins=10)
    cal_train, used_train = calibration_by_km(surv_train, y_train, lm, n_bins=10)

    # Sort by predicted survival so line segments follow left→right
    cal_val = cal_val.sort_values("pred")
    cal_train = cal_train.sort_values("pred")

    # Reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=1)

    # Train: scatter + connecting line
    ax.plot(cal_train["pred"], cal_train["obs"], '-', color='green', alpha=0.7)
    ax.scatter(cal_train["pred"], cal_train["obs"], c='green', label='Train')
    #for i, (xp, yp) in cal_test[["pred", "obs"]].iterrows():
        #ax.text(xp, yp, str(i+1), color='green', fontsize=8)

    # Validation: scatter + connecting line
    ax.plot(cal_val["pred"], cal_val["obs"], '-', color='blue', alpha=0.7)
    ax.scatter(cal_val["pred"], cal_val["obs"], c='blue', label='Validation')
    #for i, (xp, yp) in cal_val[["pred", "obs"]].iterrows():
        #ax.text(xp, yp, str(i+1), color='blue', fontsize=8)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(f"Predicted survival at {used_val:.0f} months")
    ax.set_ylabel(f"Observed survival at {used_val:.0f} months")
    ax.set_title(f"Calibration @ {used_val:.0f} months")
    ax.grid(False)
    ax.legend(loc='lower right', fontsize=8)

plt.suptitle("Calibration curves— 12, 36, 60, 120 months", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

# Make save directory
save_dir = '/Users/ryanlim/Desktop/University/IMGS/Cohort HCC Score/Model'
os.makedirs(save_dir, exist_ok=True)

# Save the network (weights + architecture)
model.save_net(os.path.join(save_dir, 'cox_model.pt'))

# Save the x_mapper
joblib.dump(x_mapper, os.path.join(save_dir, 'x_mapper.pkl'))

# Save baseline survival for later use
surv_df = model.predict_surv_df(x_train)  # or x_val
surv_df.to_csv(os.path.join(save_dir, 'baseline_survival.csv'))

print("Model and preprocessor saved.")