"""Shared definitions for the diabetes workload and the LightGBM classifier used for every cheap and expensive model.

Only the parts imported by src_extension_selective_topk_crc.py are kept: the data location, the column groups, and
train_lgb (500 trees at most, learning rate 0.05, depth 6, min_child_samples 50, early stopping after 50 rounds on the
fit split).
"""
from pathlib import Path

import lightgbm as lgb

DATA_DIR = Path("data/raw/diabetes")          # diabetic_data.csv of the UCI "Diabetes 130-US hospitals" data set

DEMOGRAPHIC_COLS = ["age", "race", "gender", "admission_type_id"]
UTILIZATION_COLS = ["number_inpatient", "number_outpatient", "number_emergency",
                    "num_medications", "num_procedures", "num_lab_procedures",
                    "time_in_hospital", "number_diagnoses"]
ALL_COLS = DEMOGRAPHIC_COLS + UTILIZATION_COLS


def train_lgb(Xtr, ytr, Xva, yva):
    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                               min_child_samples=50, verbosity=-1)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model
