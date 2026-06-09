import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import lightgbm as lgb
import joblib
import torch
from sklearn.ensemble import GradientBoostingClassifier ,HistGradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns



mpl.rcParams['agg.path.chunksize'] = 10000
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# ── Load data ─────────────────────────────────────────────────────────────────
LegSummary  = pd.read_csv("TrianningData/LegSummary_2026.csv")
FltSummary  = pd.read_csv("TrianningData/FltSummary_2026.csv")

data_FltSummary = FltSummary.copy()
data_LegSummary = LegSummary.copy()

data_LegSummary['TOTBKD'] = data_LegSummary['CurBkgs']
# ── PLF ───────────────────────────────────────────────────────────────────────
data_FltSummary['PLF'] = (data_FltSummary['RPK'] / data_FltSummary['ASK']) * 100
data_LegSummary['PLF'] = (data_LegSummary['RPK'] / data_LegSummary['ASK']) * 100

# ── Growth rates ──────────────────────────────────────────────────────────────
for df, name in [(data_FltSummary, 'Flt'), (data_LegSummary, 'Leg')]:
    df['FLTDATE'] = pd.to_datetime(df['FLTDATE'])
    for col in ['ASK', 'RPK']:
        key = f'{"ASK" if col == "ASK" else "RPK"}G_pct'
        df[key] = (
            df.groupby('FLTNUM')[col]
            .pct_change(periods=1)
            .replace([float('inf'), -float('inf')], None) * 100
        ).fillna(0).infer_objects(copy=False)

    df['HigherP'] = df['RPKG_pct'] > df['ASKG_pct']
    df['LowerP']  = df['RPKG_pct'] < df['ASKG_pct']
    df['highD']   = df['HigherP']
    df['lowD']    = df['LowerP']

# ── Overbooking flag ──────────────────────────────────────────────────────────
data_FltSummary['over'] = data_FltSummary['TOTBKD'] > data_FltSummary['RECONSTRAINEDFNLM']
print("Overbooked flights:", data_FltSummary['over'].sum())




# ── Base features ─────────────────────────────────────────────────────────────
def base_features(df):
    df['FLTDATE'] = pd.to_datetime(df['FLTDATE'])
    df['PLF']     = (df['RPK'] / df['ASK']) * 100

    df['ASKG_pct'] = (
        df.groupby('FLTNUM')['ASK']
        .pct_change(periods=1)
        .replace([float('inf'), -float('inf')], None) * 100
    ).fillna(0)

    df['RPKG_pct'] = (
        df.groupby('FLTNUM')['RPK']
        .pct_change(periods=1)
        .replace([float('inf'), -float('inf')], None) * 100
    ).fillna(0)

    # Clip extremes — values beyond ±200% are likely data artifacts
    df['RPKG_pct'] = df['RPKG_pct'].clip(-100, 200)
    df['ASKG_pct'] = df['ASKG_pct'].clip(-100, 200)

    df['highD'] = df['RPKG_pct'] > df['ASKG_pct']
    df['lowD']  = df['RPKG_pct'] < df['ASKG_pct']

    if 'TOTBKD' not in df.columns:
        df['TOTBKD'] = data_LegSummary['CurBkgs']
    return df

data_FltSummary = base_features(data_FltSummary)
data_LegSummary = base_features(data_LegSummary)


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING + 3-CLASS TARGET
# ═════════════════════════════════════════════════════════════════════════════

def prepare_ts_data(df, label=''):
    print(f"\n{'='*60}")
    print(f"  PREPARING DATA — {label}")
    print(f"{'='*60}")


    df = df.sort_values(['FLTNUM', 'FLTDATE']).reset_index(drop=True)

    flight_encoder = LabelEncoder()

    df["FLTNUM_ENC"] = (
        flight_encoder
        .fit_transform(df["FLTNUM"].astype(str))
        .astype(np.int32)
    )


    df['Month']      = df['FLTDATE'].dt.month



    # ── 3-class target ────────────────────────────────────────────────────────
    # Next period RPK growth binned into: down / flat / up
    # Thresholds: < -10% = down, -10% to +10% = flat, > +10% = up
    next_rpkg = df.groupby('FLTNUM')['RPKG_pct'].shift(-1).clip(-100, 200)

    df['TARGET'] = pd.cut(
        next_rpkg,
        bins  = [-np.inf, -10, 10, np.inf],  # tightened upper threshold: >5% = up
        labels= ['down', 'flat', 'up']
    )

    # ── Drop NaN rows ─────────────────────────────────────────────────────────
    df = df.dropna(subset=['TARGET'])
    lag_cols = [c for c in df.columns if '_lag1' in c]
    df = df.dropna(subset=lag_cols)

    # ── Filter low-record flights ─────────────────────────────────────────────
    MIN_RECORDS = 30
    counts     = df.groupby('FLTNUM')['FLTDATE'].count()
    valid_flts = counts[counts >= MIN_RECORDS].index
    df         = df[df['FLTNUM'].isin(valid_flts)]

    print(f"Flights : {df['FLTNUM'].nunique()}  |  Rows : {len(df)}")
    print(f"\nClass distribution:")
    print(df['TARGET'].value_counts())

    # ── Feature list ───────────────────────────

    feature_cols = [
        'FLTNUM',
        'Month',
        'RPK', 'PLF',
        'RPKG_pct', 'highD', 'lowD','TOTBKD'
    ]
    print(len(feature_cols))
    # ── Encode target ─────────────────────────────────────────────────────────
    le = LabelEncoder()                          # down=0, flat=1, up=2
    df['TARGET_enc'] = le.fit_transform(df['TARGET'].astype(str))
    print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    df_out = df[feature_cols + ['TARGET_enc', 'FLTDATE']].copy()

    # ── Time-based split ──────────────────────────────────────────────────────
    cutoff  = df_out['FLTDATE'].quantile(0.8)
    train   = df_out[df_out['FLTDATE'] <= cutoff].drop(columns='FLTDATE')
    test    = df_out[df_out['FLTDATE'] >  cutoff].drop(columns='FLTDATE')

    X_train = train[feature_cols].copy()
    y_train = train['TARGET_enc']
    X_test  = test[feature_cols].copy()
    y_test  = test['TARGET_enc']

    print(f"\nTrain : {len(X_train)} rows  (up to {cutoff.date()})")
    print(f"Test  : {len(X_test)} rows  (after {cutoff.date()})")

    # ── Drop NaN rows — only affects first 1-3 records per flight ──────────
    train_full = X_train.copy()
    train_full['__target__'] = y_train.values
    train_full = train_full.dropna()
    X_train = train_full.drop(columns='__target__')
    y_train  = train_full['__target__']

    test_full = X_test.copy()
    test_full['__target__'] = y_test.values
    test_full = test_full.dropna()
    X_test = test_full.drop(columns='__target__')
    y_test = test_full['__target__']

    print(f"After dropna — train: {len(X_train)} rows, test: {len(X_test)} rows")

    # ── MinMaxScaler — not affected by outliers ───────────────────────────────
    cols_to_scale = [c for c in feature_cols if c != 'FLTNUM']
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[cols_to_scale] = X_train[cols_to_scale].astype(float)
    X_test[cols_to_scale]  = X_test[cols_to_scale].astype(float)
    scaler = MinMaxScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])

    # ── Class weights instead of SMOTE — no memory overhead ──────────────────
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    class_weight = {int(c): total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sample_weight = np.array([class_weight[int(y)] for y in y_train])
    print(f"Class weights: {class_weight}")

    return X_train, X_test, y_train, y_test, feature_cols, scaler, le, sample_weight,class_weight


# ── Run for both ──────────────────────────────────────────────────────────────
X_train_flt, X_test_flt, y_train_flt, y_test_flt, features_flt, flt_scaler, flt_le, sw_flt , class_weightf = prepare_ts_data(data_FltSummary, 'FltSummary')
X_train_leg, X_test_leg, y_train_leg, y_test_leg, features_leg, leg_scaler, leg_le, sw_leg , class_weightl= prepare_ts_data(data_LegSummary, 'LegSummary')


# ═════════════════════════════════════════════════════════════════════════════
# LIGHTGBM 3-CLASS MODEL
# ═════════════════════════════════════════════════════════════════════════════

# Base monotonic constraints (33 features)
monotonic_cst_base = [
    0,  0,           # FLTNUM
    1,1,        # RPK, ASK, PLF
    1,1,1,1,# RPKG_pct, ASKG_pct

]


def make_params(feature_cols):
    monotonic_cst = monotonic_cst_base
    return {
        'objective'           : 'multiclass',
        'num_class'           : 3,
        'metric'              : 'multi_logloss',
        'learning_rate'       : 0.007,
        'num_leaves'          : 63,
        'max_depth'           : 20,
        'min_data_in_leaf'    : 50,
        'feature_fraction'    : 0.8,
        'bagging_fraction'    : 0.8,
        'bagging_freq'        : 5,
        'lambda_l1'           : 0.3,
        'lambda_l2'           : 0.8,
        'monotone_constraints': monotonic_cst,
        'verbose'             : -1,
    }


def run_model(X_train, y_train, X_test, y_test, le, label, sample_weight=None):
    print(f"\n{'='*60}")
    print(f"  MODEL — {label}")
    print(f"{'='*60}")

    params = make_params(list(X_train.columns))
    train_ds = lgb.Dataset(X_train, label=y_train, weight=sample_weight)

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(50)]
    )

    # ── Predict ───────────────────────────────────────────────────────────────
    proba  = model.predict(X_test)                    # shape (n, 3)
    y_pred = np.argmax(proba, axis=1)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n── Classification report ──")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )
    plt.title(f'Confusion Matrix — {label}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"confusion_{label}.png", dpi=150)
    plt.close()
    print(f"Confusion matrix saved → confusion_{label}.png")

    # ── Feature importance ────────────────────────────────────────────────────
    imp = pd.DataFrame({
        'feature'   : model.feature_name(),
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=imp.head(15), x='importance', y='feature', color='steelblue')
    plt.title(f'Top 15 Features — {label}')
    plt.tight_layout()
    plt.savefig(f"feature_importance_{label}.png", dpi=150)
    plt.close()
    print(f"Feature importance saved → feature_importance_{label}.png")

    print(f"\nTop 10 features:")
    print(imp.head(10).to_string(index=False))

    return model, proba


model_flt, proba_flt = run_model(X_train_flt, y_train_flt, X_test_flt, y_test_flt, flt_le, 'FltSummary', sw_flt)
model_leg, proba_leg = run_model(X_train_leg, y_train_leg, X_test_leg, y_test_leg, leg_le, 'LegSummary', sw_leg)


# ═════════════════════════════════════════════════════════════════════════════
# LOGISTIC REGRESSION — COMPARISON
# ═════════════════════════════════════════════════════════════════════════════


def run_logistic(X_train, y_train, X_test, y_test, le, label, sample_weight=None):
    print(f"\n{'='*60}")
    print(f"  LOGISTIC REGRESSION — {label}")
    print(f"{'='*60}")

    lr = LogisticRegression(
        penalty = "l2",
        solver='saga',
        max_iter=500,
        random_state=42,
        n_jobs=9,
    )
    lr.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = lr.predict(X_test)
    proba  = lr.predict_proba(X_test)


    print(f"\n── Classification report ──")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix LR — {label}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"confusion_LR_{label}.png", dpi=150)
    plt.close()
    print(f"Confusion matrix saved → confusion_LR_{label}.png")

    return lr, proba


lr_flt, lrproba_flt = run_logistic(X_train_flt, y_train_flt, X_test_flt, y_test_flt, flt_le, 'FltSummary', sw_flt)
lr_leg, lrproba_leg = run_logistic(X_train_leg, y_train_leg, X_test_leg, y_test_leg, leg_le, 'LegSummary', sw_leg)

# ═════════════════════════════════════════════════════════════════════════════
#XGBOST
# ═════════════════════════════════════════════════════════════════════════════
def run_XGBOST(X_train, y_train, X_test, y_test, le, label, sample_weight=None):
    print(f"\n{'='*60}")
    print(f"  run_XGBOST — {label}")
    print(f"{'='*60}")

    XGB = GradientBoostingClassifier(
        loss='log_loss',
        learning_rate= 0.007,
        max_depth=10,
        n_estimators=500,
        random_state=42,
        min_samples_split = 0.4,
        max_features ='log2'
    )
    XGB.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = XGB.predict(X_test)
    proba  = XGB.predict_proba(X_test)


    print(f"\n── Classification report ──")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix XGB — {label}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"confusion_XGB_{label}.png", dpi=150)
    plt.close()
    print(f"Confusion matrix saved → confusion_XGB_{label}.png")

    return XGB, proba



XGB_flt, XGBproba_flt = run_XGBOST(X_train_flt, y_train_flt, X_test_flt, y_test_flt, flt_le, 'FltSummary', sw_flt)
XGB_leg, XGBproba_leg = run_XGBOST(X_train_leg, y_train_leg, X_test_leg, y_test_leg, leg_le, 'LegSummary', sw_leg)


# ═════════════════════════════════════════════════════════════════════════════
# HistGradientBoostingClassifier
# ═════════════════════════════════════════════════════════════════════════════

def run_HGBOST(X_train, y_train, X_test, y_test, le, label,classw ,sample_weight=None):
    print(f"\n{'='*60}")
    print(f"  run_HistGradientBoostingClassifier — {label}")
    print(f"{'='*60}")
    HGB = HistGradientBoostingClassifier(
        loss='log_loss',
        learning_rate= 0.07,
        max_features=0.7,
        max_depth=25,
        random_state=42,
        validation_fraction = 0.3,
        class_weight =classw,
        l2_regularization = 1.0,
        interaction_cst = 'pairwise',
        tol = 1e-4,
        max_leaf_nodes=62,
        max_bins=128
    )
    HGB.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = HGB.predict(X_test)
    proba  = HGB.predict_proba(X_test)


    print(f"\n── Classification report ──")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix HGB — {label}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"confusion_XGB_{label}.png", dpi=150)
    plt.close()
    print(f"Confusion matrix saved → confusion_XGB_{label}.png")

    return HGB, proba



HGB_flt, HGBproba_flt = run_HGBOST(X_train_flt, y_train_flt, X_test_flt, y_test_flt, flt_le, 'FltSummary',class_weightf ,sw_flt)
HGB_leg, HGBproba_leg = run_HGBOST(X_train_leg, y_train_leg, X_test_leg, y_test_leg, leg_le, 'LegSummary',class_weightl,sw_leg)







# ═════════════════════════════════════════════════════════════════════════════
# SIDE BY SIDE COMPARISON
# ═════════════════════════════════════════════════════════════════════════════


def compare(X_test, y_test, lgb_model, lr_model,hg_model, le, label):
    print(f"\n{'='*60}")
    print(f"  COMPARISON — {label}")
    print(f"{'='*60}")

    lgb_pred = np.argmax(lgb_model.predict(X_test), axis=1)
    lr_pred  = lr_model.predict(X_test)
    X_te_t = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)
    rows = []
    for name, pred in [('LightGBM', lgb_pred), ('LogisticCV', lr_pred),('hg_model',hg_model)]:
        rows.append({
            'Model'    : name,
            'Accuracy' : round(accuracy_score(y_test, pred), 4),
            'F1_down'  : round(f1_score(y_test, pred, average=None)[0], 4),
            'F1_flat'  : round(f1_score(y_test, pred, average=None)[1], 4),
            'F1_up'    : round(f1_score(y_test, pred, average=None)[2], 4),
            'F1_macro' : round(f1_score(y_test, pred, average='macro'), 4),
        })

    cmp = pd.DataFrame(rows).set_index('Model')
    print(cmp.to_string())

    # Bar chart comparison
    cmp[['F1_down','F1_flat','F1_up','F1_macro']].plot(
        kind='bar', figsize=(8,5), colormap='Set2', edgecolor='white'
    )
    plt.title(f'LightGBM vs LogisticCV — {label}')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=0)
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"comparison_{label}.png", dpi=150)
    plt.close()
    print(f"Comparison chart saved → comparison_{label}.png")


compare(X_test_flt, y_test_flt, model_flt, lr_flt,HGB_flt, flt_le, 'FltSummary')
compare(X_test_leg, y_test_leg, model_leg, lr_leg,HGB_leg, leg_le, 'LegSummary')

# ── Save models and scalers ───────────────────────────────────────────────────
model_flt.save_model("model_flt_3class.txt")
model_leg.save_model("model_leg_3class.txt")
joblib.dump(lr_flt,"logictic_flt.plt")
joblib.dump(lr_leg,"logictic_leg.plt")
joblib.dump(HGB_flt,"HGBoost_flt.plt")
joblib.dump(HGB_leg,"HGBoost_leg.plt")
joblib.dump(flt_scaler, "flt_scaler.pkl")
joblib.dump(leg_scaler, "leg_scaler.pkl")
joblib.dump(flt_le,     "flt_label_encoder.pkl")
joblib.dump(leg_le,     "leg_label_encoder.pkl")
print("\nModels and scalers saved.")


# ═════════════════════════════════════════════════════════════════════════════
# PREDICT FUNCTION — use on new flights
# ═════════════════════════════════════════════════════════════════════════════

def predict_direction(rpk, ask, model, scaler, le, extra_features=None):
    """
    Predict RPK growth direction for a new flight record.

    Returns: predicted class (down/flat/up) + probabilities
    """
    # Build a minimal feature row — fill unknowns with 0
    row = pd.DataFrame([{f: 0 for f in model.feature_name()}])
    row['RPK'] = rpk
    row['ASK'] = ask
    row['PLF'] = (rpk / ask) * 100

    if extra_features:
        for k, v in extra_features.items():
            if k in row.columns:
                row[k] = v

    cols_to_scale = [c for c in model.feature_name() if c != 'FLTNUM']
    row[cols_to_scale] = scaler.transform(row[cols_to_scale])

    proba = model.predict(row)[0]
    pred  = le.inverse_transform([np.argmax(proba)])[0]

    print(f"\nPredicted direction : {pred.upper()}")
    print(f"P(down) = {proba[0]:.3f}  |  P(flat) = {proba[1]:.3f}  |  P(up) = {proba[2]:.3f}")
    return pred, proba


# Example
predict_direction(
    rpk=120000, ask=150000,
    model=model_flt, scaler=flt_scaler, le=flt_le,
    extra_features={'Month': 6, 'DayOfWeek': 1, 'PLF_lag1': 80}
)

































