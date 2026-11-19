"""Certified top-k with selective inference, with a finite-sample guarantee
from Conformal Risk Control (Angelopoulos, Bates, Fisch, Lei, Schuster,
ICLR 2024).

Setting. Data arrive as queries (soccer: one match; diabetes: a batch of about
500 encounters from disjoint patients; Home Credit: a batch of 1,000
applicants). In each query of n_q tuples we return k = round(k_frac * n_q)
tuples. Every tuple has a cheap estimate P_c and an expensive one P_e, and
U = P*R - (1-P)*C. The reference answer T* is the top-k under U_e (what
running the expensive model on every tuple returns).

Rule. With theta_c the midpoint between the k-th and (k+1)-th cheap scores of
the query, each tuple gets a priority computed from cheap information only:
  P0  |U_c - theta_c|
  P1  |U_c - theta_c| / (R + C)            (probability change needed)
  P2  |U_c - theta_c| / ((R + C) * s(x))   (s = predicted size of |P_e - P_c|
                                            from cheap features, P_c, R, C)
  RND random order (control)
  TOPM cheap rank / k: the simple baseline "refine the cheap top-m" with
       m = lambda * k calibrated the same way
Tuples with priority <= lambda are refined (U_c replaced by U_e); the answer
is the top-k of the mixed scores. The sets are nested in lambda.

Guarantee. The loss L(lambda) = 1 - |T_lambda ∩ T*| / k is in [0, 1] but not
always monotone in lambda (refining an over-estimated insider can let an
over-estimated outsider in). Its monotone envelope
  L~(lambda) = max over lambda' >= lambda (on the grid) of L(lambda')
is non-increasing and L <= L~. CRC on L~ over m calibration queries picks
  lambda_hat = min{ lambda : m/(m+1) * mean_cal L~(lambda) + 1/(m+1) <= eps },
which gives E[L~_test(lambda_hat)] <= eps when calibration and test queries are
exchangeable, hence E[L_test(lambda_hat)] <= eps.

Split by group (match / patient / applicant): 50% train, 10% fit (early
stopping, the scale model s and the lambda grid), 20% calibration queries,
20% test queries. For batch queries (diabetes, Home Credit) the batches are
formed on the pooled cal + test groups and a uniformly random half of them
goes to calibration (batched_cal_test), which makes the queries exactly
exchangeable. Nothing used to define the rule or the grid depends on the
calibration or test queries, which keeps them exchangeable.
Repeated over SEEDS. Existing scripts and result files are not modified.
"""
import json
import os
import sys
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.pass_success import train_success_model, predict_success_proba
from src.models.reward_model import train_reward_model, predict_reward
from src.models.utility import compute_failure_cost
from src_extension_selective_topk_homecredit import (
    CLEAN, LGD, MARGIN, UNIT, cheap_features, expensive_features, prepare,
)
from src_extension_selective_topk_diagnostic import utility
import diabetes_new_regret_diagnostic as dia  # noqa: E402  (path registered by the import above)

FRACS = {"train": 0.5, "fit": 0.1, "cal": 0.2, "test": 0.2}
K_FRACTIONS = [0.05, 0.10, 0.20]
EPSILONS = [0.05, 0.10]
PRIORITIES = ["P0", "P1", "P2", "RND", "TOPM"]
N_GRID = 200


def group_split(groups, seed):
    # Works on integer group codes: np.isin on object (string) arrays compares
    # element by element and takes hours at a million rows. Shuffling the index
    # array draws the same permutation as shuffling the unique values did, so
    # the split is unchanged.
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.arange(len(uniq))
    np.random.default_rng(seed).shuffle(order)
    cuts = np.cumsum([int(round(FRACS[s] * len(uniq))) for s in ["train", "fit", "cal"]])
    label = np.empty(len(uniq), dtype=int)
    for j, part in enumerate(np.split(order, cuts)):
        label[part] = j
    return {name: label[inv] == j for j, name in enumerate(["train", "fit", "cal", "test"])}


def batch_queries(groups, rows_per_query, seed):
    """Assign whole groups to queries of roughly rows_per_query rows.

    The leftover partial batch is merged into the previous one, so every query
    is formed the same way in the calibration and test splits.
    """
    uniq, counts = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed + 1000)
    order = rng.permutation(len(uniq))
    qid_of, acc, q = {}, 0, 0
    for i in order:
        qid_of[uniq[i]] = q
        acc += counts[i]
        if acc >= rows_per_query:
            q, acc = q + 1, 0
    if acc > 0 and q > 0:
        for g, v in qid_of.items():
            if v == q:
                qid_of[g] = q - 1
    return np.array([qid_of[g] for g in groups])


def batched_cal_test(groups, cal_mask, test_mask, rows_per_query, seed):
    """Form batch queries on the pooled cal + test groups, then send a uniformly
    random half of the queries to calibration and the rest to test.

    A uniform random permutation of one fixed list of queries is exchangeable,
    so this gives exact exchangeability (batching each split separately would
    leave one merged leftover query in each split, which is only approximately
    exchangeable). Returns (qid, cal_mask, test_mask) on the full row index.
    """
    pool = cal_mask | test_mask
    qid = np.full(len(groups), -1)
    qid[pool] = batch_queries(groups[pool], rows_per_query, seed)
    uq = np.unique(qid[pool])
    rng = np.random.default_rng(seed + 2000)
    cal_q = rng.permutation(uq)[: len(uq) // 2]
    new_cal = pool & np.isin(qid, cal_q)
    return qid, new_cal, pool & ~new_cal


# ---------------------------------------------------------------- domains
def encode(df, cols):
    """Label-free encoding done once on the whole table: categories -> fixed
    integer codes (missing -> -1), numbers kept (LightGBM handles NaN). Encoding
    each split separately would give the same category different codes in
    different splits and make a query's features depend on the other queries."""
    X = df[cols].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    return X.astype(float)


def build_soccer(seed):
    d = Path("data/processed")
    passes = pd.read_parquet(d / "passes_with_features.parquet")
    Xe = pd.read_parquet(d / "X_event.parquet").fillna(0)
    Xf = pd.read_parquet(d / "X_full.parquet").fillna(0)
    y = pd.read_parquet(d / "y_labels.parquet")["pass_success"]
    m = group_split(passes["match_id"].to_numpy(), seed)
    idx = {s: passes.index[mask] for s, mask in m.items()}
    cheap = train_success_model("lightgbm", Xe.loc[idx["train"]], y.loc[idx["train"]], Xe.loc[idx["fit"]], y.loc[idx["fit"]])
    exp = train_success_model("lightgbm", Xf.loc[idx["train"]], y.loc[idx["train"]], Xf.loc[idx["fit"]], y.loc[idx["fit"]])
    succ = passes.index[y == 1]
    reward = train_reward_model("lightgbm_regressor",
                                Xe.loc[idx["train"].intersection(succ)], passes.loc[idx["train"].intersection(succ), "delta_xt"],
                                Xe.loc[idx["fit"].intersection(succ)], passes.loc[idx["fit"].intersection(succ), "delta_xt"])
    out = {}
    for s in ["fit", "cal", "test"]:
        ix = idx[s]
        out[s] = {"pc": np.asarray(predict_success_proba(cheap, Xe.loc[ix])),
                  "pe": np.asarray(predict_success_proba(exp, Xf.loc[ix])),
                  "R": np.asarray(predict_reward(reward, Xe.loc[ix])),
                  "C": np.asarray(compute_failure_cost(passes.loc[ix], method="location_based")),
                  "xc": Xe.loc[ix].to_numpy(dtype=float), "qid": passes.loc[ix, "match_id"].to_numpy()}
    return out


_DIABETES = None


def build_diabetes(seed):
    global _DIABETES
    if _DIABETES is None:
        df = pd.read_csv(dia.DATA_DIR / "diabetic_data.csv")
        df["readmitted_30"] = (df["readmitted"] == "<30").astype(int)
        _DIABETES = (df, encode(df, dia.DEMOGRAPHIC_COLS), encode(df, dia.ALL_COLS))
    df, Xc, Xf = _DIABETES
    max_stay = df["time_in_hospital"].max()
    return build_tabular(df["patient_nbr"].to_numpy(), df["readmitted_30"].to_numpy(), Xc, Xf,
                         1.0 + df["number_inpatient"].to_numpy(dtype=float),
                         0.05 * df["time_in_hospital"].to_numpy(dtype=float) / max_stay, 500, seed)


def build_tabular(groups, y, Xc, Xf, R, C, rows_per_query, seed, positive=1):
    """Shared builder for the tabular domains: group split, batch queries with
    exact exchangeability, cheap model on Xc and expensive model on Xf (both
    trained on train, early stopped on fit). P is the probability of y == positive."""
    m = group_split(groups, seed)
    qid, m["cal"], m["test"] = batched_cal_test(groups, m["cal"], m["test"], rows_per_query, seed)
    yy = pd.Series(y)
    models = {key: dia.train_lgb(X[m["train"]], yy[m["train"]], X[m["fit"]], yy[m["fit"]])
              for key, X in {"cheap": Xc, "exp": Xf}.items()}
    col = {key: list(mod.classes_).index(positive) for key, mod in models.items()}
    out = {}
    for s in ["fit", "cal", "test"]:
        mask = m[s]
        out[s] = {"pc": models["cheap"].predict_proba(Xc[mask])[:, col["cheap"]],
                  "pe": models["exp"].predict_proba(Xf[mask])[:, col["exp"]],
                  "R": np.asarray(R, dtype=float)[mask], "C": np.asarray(C, dtype=float)[mask],
                  "xc": Xc[mask].to_numpy(dtype=float),
                  "qid": qid[mask] if s != "fit" else batch_queries(groups[mask], rows_per_query, seed)}
    return out


_HC = None


def build_home_credit(seed):
    global _HC
    if _HC is None:
        con = duckdb.connect()
        prepare(con)
        app = con.execute(f"select * from '{CLEAN / 'application_train_clean.parquet'}'").df()
        cache = CLEAN / "agg_all_clean.parquet"
        if cache.exists():
            agg = pd.read_parquet(cache)
        else:
            agg = expensive_features(con, app["SK_ID_CURR"].to_numpy())
            agg.to_parquet(cache)
        Xc = cheap_features(app).reset_index(drop=True)
        Xf = pd.concat([Xc, agg.loc[app["SK_ID_CURR"].to_numpy()].reset_index(drop=True)], axis=1)
        _HC = (app, Xc, Xf)
    app, Xc, Xf = _HC
    m = group_split(app["SK_ID_CURR"].to_numpy(), seed)
    qid, m["cal"], m["test"] = batched_cal_test(app["SK_ID_CURR"].to_numpy(), m["cal"], m["test"], 1000, seed)
    y = app["TARGET"]
    models = {key: dia.train_lgb(X[m["train"]], y[m["train"]], X[m["fit"]], y[m["fit"]])
              for key, X in {"cheap": Xc, "exp": Xf}.items()}
    out = {}
    for s in ["fit", "cal", "test"]:
        mask = m[s]
        credit = app.loc[mask, "AMT_CREDIT"].to_numpy(dtype=float) / UNIT
        out[s] = {"pc": 1 - models["cheap"].predict_proba(Xc[mask])[:, 1],
                  "pe": 1 - models["exp"].predict_proba(Xf[mask])[:, 1],
                  "R": MARGIN * credit, "C": LGD * credit,
                  "xc": Xc[mask].to_numpy(dtype=float),
                  "qid": qid[mask] if s != "fit" else batch_queries(app.loc[mask, "SK_ID_CURR"].to_numpy(), 1000, seed)}
    return out


_KDD = None
KDD_RAW = Path("data/raw/kddcup98/cup98LRN.txt")
KDD_MAIL_COST = 0.68
# Giving and promotion history (the "expensive" join against the donation ledger);
# everything else in the record is mailing-list demographics and census data.
KDD_HISTORY_PREFIXES = ("ADATE_", "RFA_", "RDATE_", "RAMNT_", "MDMAUD")
KDD_HISTORY = ["CARDPROM", "MAXADATE", "NUMPROM", "CARDPM12", "NUMPRM12", "RAMNTALL", "NGIFTALL", "CARDGIFT",
               "MINRAMNT", "MINRDATE", "MAXRAMNT", "MAXRDATE", "LASTGIFT", "LASTDATE", "FISTDATE", "NEXTDATE",
               "TIMELAG", "AVGGIFT"]


def build_kdd98(seed):
    """KDD Cup 1998 donor mailing (UCI). P = response probability, D = gift.
    Cheap: demographics + census. Expensive: + 24-promotion giving history.
    Profit of mailing = P * D - 0.68, i.e. R = D_hat - 0.68, C = 0.68, where
    D_hat is predicted from cheap features by a regressor fit on the training
    split's responders (so R is known without the expensive model)."""
    global _KDD
    if _KDD is None:
        df = pd.read_csv(KDD_RAW, low_memory=False)
        hist = [c for c in df.columns if c.startswith(KDD_HISTORY_PREFIXES) or c in KDD_HISTORY]
        cheap = [c for c in df.columns if c not in hist and c not in ("CONTROLN", "TARGET_B", "TARGET_D", "ZIP")]
        _KDD = (df, encode(df, cheap), encode(df, cheap + hist))
    df, Xc, Xf = _KDD
    groups = df["CONTROLN"].to_numpy()
    train = group_split(groups, seed)["train"] & (df["TARGET_B"].to_numpy() == 1)
    reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                            verbosity=-1, random_state=42).fit(Xc[train], df.loc[train, "TARGET_D"])
    d_hat = np.maximum(reg.predict(Xc), 1.0)
    return build_tabular(groups, df["TARGET_B"].to_numpy(), Xc, Xf, d_hat - KDD_MAIL_COST,
                         np.full(len(df), KDD_MAIL_COST), 250, seed)


_TELCO = None
TELCO_RAW = Path("data/raw/telco_churn/Telco-Customer-Churn.csv")
TELCO_ACCOUNT = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "Contract", "PaperlessBilling",
                 "PaymentMethod", "MonthlyCharges", "TotalCharges"]
TELCO_SERVICES = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                  "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
TELCO_KEEP_MONTHS, TELCO_OFFER_MONTHS = 12, 2


def build_telco(seed):
    """IBM Telco customer churn. P = churn probability. Retention offer: a
    churner who is kept brings 12 months of charges (R), the offer given to a
    non-churner wastes 2 months of charges (C). Cheap: account and billing
    table. Expensive: + subscribed-services table."""
    global _TELCO
    if _TELCO is None:
        df = pd.read_csv(TELCO_RAW)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        _TELCO = (df, encode(df, TELCO_ACCOUNT), encode(df, TELCO_ACCOUNT + TELCO_SERVICES))
    df, Xc, Xf = _TELCO
    charge = df["MonthlyCharges"].to_numpy(dtype=float)
    return build_tabular(df["customerID"].to_numpy(), (df["Churn"] == "Yes").astype(int).to_numpy(), Xc, Xf,
                         TELCO_KEEP_MONTHS * charge, TELCO_OFFER_MONTHS * charge, 50, seed)


_IEEE = None
IEEE_RAW = Path("data/raw/ieee_fraud")
IEEE_CLEAN = Path("data/processed/ieee_fraud/train_clean.parquet")
IEEE_REVIEW_COST = 10.0


def build_ieee_fraud(seed):
    """IEEE-CIS fraud detection (Kaggle, Vesta). P = fraud probability.
    Reviewing a transaction stops a fraud worth its amount (R = TransactionAmt)
    and costs IEEE_REVIEW_COST when the transaction is legitimate (C).
    Cheap: core transaction fields (amount, card, address, e-mail domains,
    C/D/M counters). Expensive: + 339 Vesta V features and the identity table
    (device and network fields), joined on TransactionID. Groups approximate a
    client as card1 + addr1 + (transaction day - D1), the usual identification
    for this data, so that one client's transactions stay in one split."""
    global _IEEE
    if _IEEE is None:
        if not IEEE_CLEAN.exists():
            IEEE_CLEAN.parent.mkdir(parents=True, exist_ok=True)
            t = pd.read_csv(IEEE_RAW / "train_transaction.csv")
            i = pd.read_csv(IEEE_RAW / "train_identity.csv")
            t.merge(i, on="TransactionID", how="left").to_parquet(IEEE_CLEAN)
        df = pd.read_parquet(IEEE_CLEAN)
        cheap = (["TransactionAmt", "ProductCD", "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain"]
                 + [f"card{j}" for j in range(1, 7)] + [f"C{j}" for j in range(1, 15)]
                 + [f"D{j}" for j in range(1, 16)] + [f"M{j}" for j in range(1, 10)])
        exp = cheap + [c for c in df.columns if c.startswith(("V", "id_")) or c in ("DeviceType", "DeviceInfo")]
        day = (np.floor(df["TransactionDT"] / 86400) - df["D1"]).fillna(-1)
        groups = (df["card1"].astype(str) + "_" + df["addr1"].fillna(-1).astype(str) + "_" + day.astype(str)).to_numpy(dtype=object)
        _IEEE = (df[["isFraud", "TransactionAmt"]].copy(), groups,
                 encode(df, cheap).astype(np.float32), encode(df, exp).astype(np.float32))
        del df
    df, groups, Xc, Xf = _IEEE
    return build_tabular(groups, df["isFraud"].to_numpy(), Xc, Xf, df["TransactionAmt"].to_numpy(dtype=float),
                         np.full(len(df), IEEE_REVIEW_COST), 1000, seed)


_LC = None
LC_RAW = Path("data/raw/lending_club/accepted_2007_to_2018Q4.csv.gz")
LC_CLEAN = Path("data/processed/lending_club/accepted_clean.parquet")
LC_LISTING = ["loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade", "emp_length", "home_ownership",
              "annual_inc", "verification_status", "purpose", "addr_state", "dti", "application_type",
              "initial_list_status"]
LC_BUREAU = ["fico_range_low", "fico_range_high", "earliest_cr_line", "inq_last_6mths", "delinq_2yrs",
             "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec", "revol_bal", "revol_util",
             "total_acc", "collections_12_mths_ex_med", "mths_since_last_major_derog", "acc_now_delinq",
             "tot_coll_amt", "tot_cur_bal", "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m",
             "mths_since_rcnt_il", "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util",
             "total_rev_hi_lim", "inq_fi", "total_cu_tl", "inq_last_12m", "acc_open_past_24mths", "avg_cur_bal",
             "bc_open_to_buy", "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
             "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc", "mths_since_recent_bc",
             "mths_since_recent_bc_dlq", "mths_since_recent_inq", "mths_since_recent_revol_delinq",
             "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
             "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats", "num_tl_120dpd_2m",
             "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75",
             "pub_rec_bankruptcies", "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
             "total_il_high_credit_limit"]


def build_lending_club(seed):
    """Lending Club accepted loans 2007-2018Q4 (Kaggle mirror), finished loans
    only (fully paid or charged off). P = repayment probability.
    R = interest earned if repaid (installment * term - loan amount),
    C = LGD * loan amount, with LGD the mean loss rate of the training split's
    charged-off loans. Cheap: what the listing shows. Expensive: + the full
    credit bureau report (a paid pull in practice). Post-origination fields
    (payments, recoveries, last FICO, hardship, settlement) are never features."""
    global _LC
    if _LC is None:
        if not LC_CLEAN.exists():
            LC_CLEAN.parent.mkdir(parents=True, exist_ok=True)
            cols = ["id", "loan_status", "total_rec_prncp", "recoveries"] + LC_LISTING + LC_BUREAU
            raw = pd.read_csv(LC_RAW, usecols=cols, low_memory=False)
            status = raw["loan_status"].astype(str).str.replace("Does not meet the credit policy. Status:", "")
            raw = raw[status.isin(["Fully Paid", "Charged Off"])].copy()
            raw["repaid"] = (status[raw.index] == "Fully Paid").astype(int)
            raw["term"] = raw["term"].astype(str).str.extract(r"(\d+)")[0].astype(float)
            raw["earliest_cr_line"] = pd.to_datetime(raw["earliest_cr_line"], format="%b-%Y", errors="coerce").dt.year
            raw.drop(columns="loan_status").reset_index(drop=True).to_parquet(LC_CLEAN)
        df = pd.read_parquet(LC_CLEAN)
        _LC = (df[["id", "repaid", "loan_amnt", "installment", "term", "total_rec_prncp", "recoveries"]].copy(),
               encode(df, LC_LISTING).astype(np.float32), encode(df, LC_LISTING + LC_BUREAU).astype(np.float32))
        del df
    df, Xc, Xf = _LC
    groups = df["id"].astype(str).to_numpy(dtype=object)
    amt = df["loan_amnt"].to_numpy(dtype=float)
    off = group_split(groups, seed)["train"] & (df["repaid"].to_numpy() == 0)
    lgd = float(np.clip((amt - df["total_rec_prncp"].to_numpy() - df["recoveries"].to_numpy())[off] / amt[off], 0, 1).mean())
    R = np.maximum(df["installment"].to_numpy() * df["term"].to_numpy() - amt, 0.0)
    return build_tabular(groups, df["repaid"].to_numpy(), Xc, Xf, R, lgd * amt, 5000, seed)


# CERTTOPK_LLM_SCORES selects another scoring run of the same loans (e.g. the deterministic concurrency-1 run)
LLM_SCORES = Path(os.environ.get("CERTTOPK_LLM_SCORES", "paper/src_extension/selective_topk_llm_scores.csv"))
LC_LLM_ROWS = 500
_LCL = None


def lc_llm_frame():
    """The Lending Club loans the local LLM scored (finished loans with a borrower
    description longer than 30 characters, 2007-2014), in a fixed row order, with
    the LLM score as a logit."""
    global _LCL
    if _LCL is None:
        df = pd.read_parquet(LC_CLEAN)
        sc = pd.read_csv(LLM_SCORES).drop_duplicates("id")
        df["id"], sc["id"] = df["id"].astype(str), sc["id"].astype(str)
        df = df.merge(sc[["id", "p_llm_yes"]], on="id", how="inner").reset_index(drop=True)
        p = df["p_llm_yes"].clip(1e-6, 1 - 1e-6)
        df["llm_logit"] = np.log(p / (1 - p))
        _LCL = df
    return _LCL


def build_lc_llm(seed):
    """Lending Club with a local LLM as the expensive model. Same loans as
    lc_llm_frame. Cheap: what the listing shows (LC_LISTING). Expensive: + the
    LLM's reading of the borrower's free-text description (Qwen2.5-7B-Instruct
    Q4 on an RTX 5060 Ti, P(Yes) to "will this borrower fully repay", entered as
    a logit), i.e. one GPU call per loan (about 0.055 s vs under a microsecond for
    the cheap model; src_extension_selective_topk_llm_trial.py). R and C as in
    build_lending_club. Queries of LC_LLM_ROWS loans."""
    df = lc_llm_frame()
    Xc = encode(df, LC_LISTING).astype(np.float32)
    Xf = encode(df, LC_LISTING + ["llm_logit"]).astype(np.float32)
    groups = df["id"].to_numpy(dtype=object)
    amt = df["loan_amnt"].to_numpy(dtype=float)
    off = group_split(groups, seed)["train"] & (df["repaid"].to_numpy() == 0)
    lgd = float(np.clip((amt - df["total_rec_prncp"].to_numpy() - df["recoveries"].to_numpy())[off] / amt[off], 0, 1).mean())
    R = np.maximum(df["installment"].to_numpy() * df["term"].to_numpy() - amt, 0.0)
    return build_tabular(groups, df["repaid"].to_numpy(), Xc, Xf, R, lgd * amt, LC_LLM_ROWS, seed)


CACHE = Path("data/processed/selective_topk_cache")


def cached(name, build):
    """The large domains take minutes per seed to build (two LightGBM models on
    hundreds of thousands of rows). The build is deterministic in the seed, so
    the CRC, LTT and Everest scripts share one saved copy per (domain, seed)."""
    def f(seed):
        path = CACHE / f"{name}_seed{seed}_clean.npz"
        if path.exists():
            z = np.load(path, allow_pickle=True)
            return {s: {key: z[f"{s}__{key}"] for key in ["pc", "pe", "R", "C", "xc", "qid"]} for s in ["fit", "cal", "test"]}
        data = build(seed)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{f"{s}__{key}": v for s, d in data.items() for key, v in d.items()})
        return data
    return f


DOMAINS = {"soccer": build_soccer, "diabetes": build_diabetes, "home_credit": build_home_credit,
           "kdd98": build_kdd98, "telco": build_telco, "ieee_fraud": cached("ieee_fraud", build_ieee_fraud),
           "lending_club": cached("lending_club", build_lending_club), "lc_llm": cached("lc_llm", build_lc_llm)}


# ---------------------------------------------------------------- method
def fit_scale(fit):
    X = np.column_stack([fit["xc"], fit["pc"], fit["R"], fit["C"]])
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                              verbosity=-1, random_state=42)
    model.fit(X, np.abs(fit["pe"] - fit["pc"]))
    floor = float(np.quantile(np.abs(fit["pe"] - fit["pc"]), 0.05)) + 1e-6
    return lambda d: np.maximum(model.predict(np.column_stack([d["xc"], d["pc"], d["R"], d["C"]])), floor)


def query_arrays(d, scale, k_frac, seed, stream=None):
    """Per query: U_c, U_e, k, and the priorities.

    stream separates the random (RND) priorities of the fit / calibration / test
    splits (0 / 1 / 2): with one shared stream, row i of the calibration split
    and row i of the test split would get the same random number, which breaks
    exact exchangeability of the RND rule. stream=None keeps the single stream
    used by the Pareto script, which only uses the test split.
    """
    U_c, U_e = utility(d["pc"], d["R"], d["C"]), utility(d["pe"], d["R"], d["C"])
    assert np.all(d["R"] + d["C"] > 0), "the priorities divide by the budget R + C"
    s = scale(d)
    rng = np.random.default_rng(seed + 7 if stream is None else [seed, 7, stream])
    rnd = rng.random(len(U_c))
    queries = []
    for q in np.unique(d["qid"]):
        ix = np.flatnonzero(d["qid"] == q)
        n_q = len(ix)
        k = max(1, int(round(k_frac * n_q)))
        if k >= n_q:
            continue
        uc = U_c[ix]
        part = -np.partition(-uc, [k - 1, k])
        dist = np.abs(uc - 0.5 * (part[k - 1] + part[k]))
        budget = d["R"][ix] + d["C"][ix]
        rank = np.empty(n_q)
        rank[np.argsort(-uc, kind="stable")] = np.arange(n_q)
        queries.append({"uc": uc, "ue": U_e[ix], "k": k, "pri": {
            "P0": dist, "P1": dist / budget, "P2": dist / (budget * s[ix]), "RND": rnd[ix], "TOPM": rank / k}})
    return queries


def topk(score, k):
    """Indices of the k largest scores; ties at the k-th score are broken by the
    smaller candidate index, i.e. lexicographic order on (-score, index).
    (np.argpartition alone leaves the choice among tied scores unspecified.)"""
    kth = -np.partition(-score, k - 1)[k - 1]
    above = np.flatnonzero(score > kth)
    tied = np.flatnonzero(score == kth)
    return np.concatenate([above, tied[: k - len(above)]])


def loss_curve(qr, name, grid):
    """L(lambda) and refined fraction on the grid for one query."""
    ref = set(topk(qr["ue"], qr["k"]).tolist())
    pri = qr["pri"][name]
    L, frac = np.empty(len(grid)), np.empty(len(grid))
    for j, lam in enumerate(grid):
        refined = pri <= lam
        mixed = np.where(refined, qr["ue"], qr["uc"])
        L[j] = 1 - len(ref & set(topk(mixed, qr["k"]).tolist())) / qr["k"]
        frac[j] = refined.mean()
    return L, frac


def envelope(L):
    return np.maximum.accumulate(L[::-1])[::-1]


def run(domain, seed):
    data = DOMAINS[domain](seed)
    scale = fit_scale(data["fit"])
    rows = []
    for k_frac in K_FRACTIONS:
        fit = query_arrays(data["fit"], scale, k_frac, seed, stream=0)
        cal = query_arrays(data["cal"], scale, k_frac, seed, stream=1)
        test = query_arrays(data["test"], scale, k_frac, seed, stream=2)
        m = len(cal)
        for name in PRIORITIES:
            # The grid must not depend on calibration or test data (exchangeability), so it comes from the fit split.
            pooled = np.concatenate([q["pri"][name] for q in fit])
            grid = np.unique(np.concatenate([[-np.inf], np.quantile(pooled, np.linspace(0, 1, N_GRID + 1)), [np.inf]]))
            Lt_cal = np.array([envelope(loss_curve(q, name, grid)[0]) for q in cal])
            test_curves = [loss_curve(q, name, grid) for q in test]
            L_test = np.array([c[0] for c in test_curves])
            Lt_test = np.array([envelope(c[0]) for c in test_curves])
            F_test = np.array([c[1] for c in test_curves])
            for eps in EPSILONS:
                ok = np.flatnonzero(m / (m + 1) * Lt_cal.mean(axis=0) + 1 / (m + 1) <= eps)
                j = ok[0] if len(ok) else len(grid) - 1
                rows.append({"domain": domain, "seed": seed, "k_fraction": k_frac, "eps": eps, "priority": name,
                             "m_cal": m, "n_test_queries": len(test), "lambda_index": int(j),
                             "test_miss": float(L_test[:, j].mean()), "test_miss_envelope": float(Lt_test[:, j].mean()),
                             "test_call_fraction": float(F_test[:, j].mean())})
    return rows


def kept_rows(path, domains):
    """Rows of an earlier results file for the domains not being rerun, so that
    running a subset of domains adds to the file instead of replacing it."""
    if len(sys.argv) <= 2 or not path.exists():
        return pd.DataFrame()
    old = pd.read_csv(path)
    return old[~old["domain"].isin(domains)]


def main():
    seeds = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 20))
    domains = sys.argv[2].split(",") if len(sys.argv) > 2 else list(DOMAINS)
    out_dir = Path("paper/src_extension")
    keep = kept_rows(out_dir / "selective_topk_crc_results.csv", domains)
    rows = []
    for domain in domains:
        for seed in seeds:
            t0 = time.time()
            rows += run(domain, seed)
            print(f"[{domain}] seed {seed} done in {time.time() - t0:.0f}s", flush=True)
            pd.concat([keep, pd.DataFrame(rows)]).to_csv(out_dir / "selective_topk_crc_results.csv", index=False)
    t = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)
    t.to_json(out_dir / "selective_topk_crc_results.json", orient="records", indent=2)
    summ = t.groupby(["domain", "k_fraction", "eps", "priority"]).agg(
        seeds=("seed", "nunique"), m_cal=("m_cal", "mean"),
        miss_mean=("test_miss", "mean"), miss_sd=("test_miss", "std"), miss_max=("test_miss", "max"),
        seeds_over_eps=("test_miss", lambda s: int((s > t.loc[s.index, "eps"]).sum())),
        calls_mean=("test_call_fraction", "mean")).reset_index()
    # Seed-level interval (seeds re-split the same data, so the seed is the unit): t with seeds - 1 df.
    from scipy.stats import t as student_t
    summ["miss_ci95_hi"] = summ["miss_mean"] + student_t.ppf(0.975, summ["seeds"] - 1) * summ["miss_sd"] / np.sqrt(summ["seeds"])
    summ.to_csv(out_dir / "selective_topk_crc_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
