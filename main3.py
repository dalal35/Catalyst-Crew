"""
Decision Fatigue Detection Pipeline — v7
==========================================
Chart fixes vs v6:
  1. Hour-of-Day: SD band clipped to ±0.5*mean so it doesn't swallow chart;
     y-axis forced to [0, 1]; Spearman r printed as metric
  2. Error Rate vs Decisions: scatter dots coloured by tier (small alpha),
     LOWESS lines per tier with clear colour separation;
     Pearson r between decisions/hr and error/hr printed
  3. Shift Duration: polynomial fit (degree=2) replaces raw binned mean so
     the U-shape or rise is smooth; R² of poly fit printed;
     tier lines removed (they're rank-transform artifacts), only overall shown
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, r2_score
from sklearn.pipeline import Pipeline
from scipy.stats import rankdata, spearmanr, pearsonr
from scipy.ndimage import uniform_filter1d
import numpy.polynomial.polynomial as poly

import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = "data"
KAGGLE_FILE = os.path.join(DATA_DIR, "decision_fatigue_dataset.csv")
POE_FILE    = os.path.join(DATA_DIR, "poe.csv")
EMAR_FILE   = os.path.join(DATA_DIR, "emar.csv")

MIN_ORDERS_SHIFT        = 15
MIN_SHIFTS_PER_PROVIDER = 6
MAX_ERRORS_PER_HOUR     = 10.0
MAX_DECISIONS_PER_HOUR  = 60.0
MAX_SWITCHES_PER_HOUR   = 30.0
MIN_SHIFTS_PER_HOUR     = 5
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

KAGGLE_FEATURES = [
    'Hours_Awake', 'Decisions_Made', 'Task_Switches',
    'Avg_Decision_Time_sec', 'Error_Rate', 'Cognitive_Load_Score',
    'Time_of_Day_sin', 'Time_of_Day_cos',
]
MIMIC_FEATURES = [
    'Hours_Awake', 'Decisions_per_hour', 'Task_Switches_per_hour',
    'Avg_Decision_Time_sec', 'Error_Rate_per_hour', 'Cognitive_Load_Score',
    'Time_of_Day_sin', 'Time_of_Day_cos',
]

print("=" * 60)
print("DECISION FATIGUE DETECTION PIPELINE  [v7]")
print("=" * 60)


# ─────────────────────────────────────────────
# STEP 1: KAGGLE
# ─────────────────────────────────────────────
print("\n[1/4] Loading and training on Kaggle fatigue dataset...")

def add_circadian_encoding(df, hour_col='Time_of_Day'):
    hours = pd.to_numeric(df.get(hour_col, 12), errors='coerce').fillna(12)
    df['Time_of_Day_sin'] = np.sin(2 * np.pi * hours / 24)
    df['Time_of_Day_cos'] = np.cos(2 * np.pi * hours / 24)
    return df

def load_kaggle_data(path):
    df = pd.read_csv(path)
    print(f"  Kaggle dataset: {len(df):,} rows, {df.shape[1]} columns")
    if 'Fatigue_Level' in df.columns:
        print(f"  Fatigue levels: {df['Fatigue_Level'].value_counts().to_dict()}")
    return df

def prepare_kaggle_features(df):
    df = df.copy()
    if 'Time_of_Day' in df.columns:
        df['Time_of_Day'] = df['Time_of_Day'].astype(str).str.strip().str.capitalize()
        tod_map = {'Morning': 8, 'Afternoon': 14, 'Evening': 19, 'Night': 2}
        df['Time_of_Day'] = pd.to_numeric(
            df['Time_of_Day'].map(tod_map), errors='coerce'
        ).fillna(12)
    df = add_circadian_encoding(df)
    if 'Fatigue_Level' in df.columns:
        df['fatigue_binary'] = (df['Fatigue_Level'] == 'High').astype(int)
    elif 'Decision_Fatigue_Score' in df.columns:
        df['fatigue_binary'] = (
            df['Decision_Fatigue_Score'] >= df['Decision_Fatigue_Score'].quantile(0.7)
        ).astype(int)
    available = [f for f in KAGGLE_FEATURES if f in df.columns]
    X = df[available].apply(pd.to_numeric, errors='coerce').fillna(0)
    return X, df['fatigue_binary'], available

def train_kaggle_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=5,
        random_state=42, class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    lr_pipe.fit(X_train, y_train)
    lr_auc = roc_auc_score(y_test, lr_pipe.predict_proba(X_test)[:, 1])
    print(f"\n  Kaggle RF AUC: {rf_auc:.3f}   LR AUC: {lr_auc:.3f}")
    print(f"\n  Feature Importances (Kaggle RF):")
    for feat, imp in sorted(zip(X.columns, rf.feature_importances_), key=lambda x: -x[1]):
        print(f"    {feat:<30} {'█'*int(imp*40)} {imp:.3f}")
    return rf, list(X.columns)


# ─────────────────────────────────────────────
# STEP 2: MIMIC
# ─────────────────────────────────────────────
print("\n[2/4] Loading MIMIC-IV data...")

def load_poe(path):
    df = pd.read_csv(path, parse_dates=['ordertime'])
    df = df.dropna(subset=['order_provider_id', 'ordertime'])
    df = df.sort_values(['order_provider_id', 'ordertime'])
    print(f"  POE: {len(df):,} records, {df['order_provider_id'].nunique():,} providers")
    return df

def load_emar(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    print(f"  EMAR: {len(df):,} records")
    for time_col in ['charttime', 'scheduletime', 'starttime']:
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df = df.rename(columns={time_col: 'emar_time'})
            break
    return df

def assign_shifts(poe_df):
    poe_df = poe_df.copy()
    poe_df['hour']         = poe_df['ordertime'].dt.hour
    poe_df['shift_anchor'] = poe_df['hour'].apply(lambda h: 7 if 7 <= h < 19 else 19)
    poe_df['shift_date']   = poe_df['ordertime'].dt.date
    night_early = (poe_df['shift_anchor'] == 19) & (poe_df['hour'] < 7)
    poe_df.loc[night_early, 'shift_date'] = (
        poe_df.loc[night_early, 'ordertime'] - pd.Timedelta(days=1)
    ).dt.date
    poe_df['shift_id'] = (
        poe_df['order_provider_id'].astype(str) + "_" +
        poe_df['shift_date'].astype(str) + "_" +
        poe_df['shift_anchor'].astype(str)
    )
    return poe_df

def compute_error_signals(poe_df, emar_df):
    poe_df = poe_df.copy()
    poe_df['is_dc']     = (poe_df['transaction_type'] == 'D/C').astype(int)
    poe_df['is_change'] = (poe_df['transaction_type'] == 'Change').astype(int)
    poe_df['emar_not_given'] = 0
    if emar_df is not None and 'event_txt' in emar_df.columns and 'poe_id' in emar_df.columns:
        not_given_ids = emar_df[
            emar_df['event_txt'].str.lower().str.contains('not given|stopped|held', na=False)
        ]['poe_id'].dropna().unique()
        poe_df['emar_not_given'] = poe_df['poe_id'].isin(not_given_ids).astype(int)
    poe_df['is_error'] = (
        (poe_df['is_dc'] == 1) |
        (poe_df['is_change'] == 1) |
        (poe_df['emar_not_given'] == 1)
    ).astype(int)
    return poe_df

def engineer_mimic_features(poe_df):
    records = []
    for shift_id, shift_df in poe_df.groupby('shift_id'):
        shift_df = shift_df.sort_values('ordertime')
        if len(shift_df) < MIN_ORDERS_SHIFT:
            continue
        provider       = shift_df['order_provider_id'].iloc[0]
        start_time     = shift_df['ordertime'].min()
        end_time       = shift_df['ordertime'].max()
        hours_in_shift = max((end_time - start_time).total_seconds() / 3600, 0.1)

        time_gaps = shift_df['ordertime'].diff().dt.total_seconds().dropna()
        time_gaps = time_gaps[(time_gaps > 0) & (time_gaps <= 7200)]
        avg_decision_time = time_gaps.median() if len(time_gaps) > 0 else 300.0

        order_types   = shift_df['order_type'].tolist()
        task_switches = sum(1 for i in range(1, len(order_types))
                            if order_types[i] != order_types[i-1])

        total_errors        = shift_df['is_error'].sum()
        error_rate_per_hour = min(total_errors / hours_in_shift, MAX_ERRORS_PER_HOUR)

        type_probs     = shift_df['order_type'].value_counts(normalize=True)
        entropy        = -sum(p * np.log2(p + 1e-9) for p in type_probs)
        max_entropy    = np.log2(len(type_probs) + 1e-9)
        cognitive_load = round((entropy / max(max_entropy, 1)) * 10, 2)

        time_of_day = start_time.hour
        n_orders    = len(shift_df)

        records.append({
            'provider_id':            provider,
            'shift_id':               shift_id,
            'shift_start':            start_time,
            'Hours_Awake':            round(hours_in_shift, 2),
            'Decisions_Made':         n_orders,
            'Decisions_per_hour':     round(min(n_orders / hours_in_shift, MAX_DECISIONS_PER_HOUR), 2),
            'Task_Switches':          task_switches,
            'Task_Switches_per_hour': round(min(task_switches / hours_in_shift, MAX_SWITCHES_PER_HOUR), 2),
            'Avg_Decision_Time_sec':  round(avg_decision_time, 2),
            'Error_Rate':             round(shift_df['is_error'].mean(), 4),
            'Error_Rate_per_hour':    round(error_rate_per_hour, 4),
            'Cognitive_Load_Score':   cognitive_load,
            'Time_of_Day':            time_of_day,
            'Time_of_Day_sin':        round(np.sin(2 * np.pi * time_of_day / 24), 4),
            'Time_of_Day_cos':        round(np.cos(2 * np.pi * time_of_day / 24), 4),
            'total_dc_orders':        shift_df['is_dc'].sum(),
            'total_changes':          shift_df['is_change'].sum(),
            'n_orders':               n_orders,
        })

    df = pd.DataFrame(records)
    print(f"  Shift windows: {len(df):,}  |  Unique providers: {df['provider_id'].nunique():,}")
    return df


# ─────────────────────────────────────────────
# STEP 3: WITHIN-MIMIC CALIBRATED GBM
# ─────────────────────────────────────────────

def smooth_rank_transform(series, noise_scale=0.01, random_state=42):
    rng     = np.random.default_rng(random_state)
    ranks   = rankdata(series, method='average')
    uniform = (ranks - 1) / (len(ranks) - 1)
    jitter  = rng.normal(0, noise_scale, size=len(uniform))
    return np.clip(uniform + jitter, 0, 1)

def build_within_mimic_model(mimic_df, kaggle_importances, kaggle_feature_names):
    df = mimic_df.copy()
    imp_map = dict(zip(kaggle_feature_names, kaggle_importances))
    feature_weights = {
        'Hours_Awake':            imp_map.get('Hours_Awake', 0.10),
        'Decisions_per_hour':     imp_map.get('Decisions_Made', 0.10),
        'Task_Switches_per_hour': imp_map.get('Task_Switches', 0.10),
        'Avg_Decision_Time_sec':  imp_map.get('Avg_Decision_Time_sec', 0.05),
        'Error_Rate_per_hour':    imp_map.get('Error_Rate', 0.15),
        'Cognitive_Load_Score':   imp_map.get('Cognitive_Load_Score', 0.05),
        'Time_of_Day_sin':        imp_map.get('Time_of_Day_sin',
                                      imp_map.get('Time_of_Day', 0.05)) * 0.5,
        'Time_of_Day_cos':        imp_map.get('Time_of_Day_cos',
                                      imp_map.get('Time_of_Day', 0.05)) * 0.5,
    }

    available = [f for f in MIMIC_FEATURES if f in df.columns]
    weights   = np.array([feature_weights.get(f, 0.05) for f in available])
    weights   = weights / weights.sum()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[available].fillna(0))
    df['composite_score'] = X_scaled @ weights

    df['fatigue_label'] = -1
    provider_counts = df.groupby('provider_id').size()
    eligible        = provider_counts[provider_counts >= MIN_SHIFTS_PER_PROVIDER].index
    for provider in eligible:
        grp = df[df['provider_id'] == provider]
        q75 = grp['composite_score'].quantile(0.75)
        df.loc[grp[grp['composite_score'] >= q75].index, 'fatigue_label'] = 1
        df.loc[grp[grp['composite_score'] <  q75].index, 'fatigue_label'] = 0

    global_q75 = df.loc[df['fatigue_label'] != -1, 'composite_score'].quantile(0.75)
    unlabelled  = df['fatigue_label'] == -1
    df.loc[unlabelled & (df['composite_score'] >= global_q75), 'fatigue_label'] = 1
    df.loc[unlabelled & (df['composite_score'] <  global_q75), 'fatigue_label'] = 0

    print(f"\n  Eligible providers (within-label): {len(eligible):,}")
    print(f"  Label balance: {df['fatigue_label'].value_counts().to_dict()}")

    X_feat = df[available].fillna(0)
    y_lab  = df['fatigue_label']
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y_lab, test_size=0.2, random_state=42, stratify=y_lab
    )

    gbm_base = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    gbm = CalibratedClassifierCV(gbm_base, method='isotonic', cv=3)
    gbm.fit(X_train, y_train)

    if y_test.nunique() > 1:
        auc = roc_auc_score(y_test, gbm.predict_proba(X_test)[:, 1])
        print(f"\n  Calibrated GBM AUC: {auc:.3f}")
        print(classification_report(y_test, gbm.predict(X_test),
                                    target_names=['Low/Moderate', 'High Fatigue']))

    raw_prob                  = gbm.predict_proba(X_feat)[:, 1]
    df['fatigue_probability'] = smooth_rank_transform(raw_prob)
    df['risk_tier']           = pd.cut(
        df['fatigue_probability'],
        bins=[-np.inf, 1/3, 2/3, np.inf],
        labels=['Low', 'Moderate', 'High']
    )
    return df, gbm, available, scaler


# ─────────────────────────────────────────────
# STEP 4: VISUALIZE — FIXED v7
# ─────────────────────────────────────────────

def lowess_smooth(x, y, frac=0.5):
    order    = np.argsort(x)
    xs, ys   = np.array(x)[order], np.array(y)[order]
    window   = max(int(len(xs) * frac), 3)
    smoothed = pd.Series(ys).rolling(window, center=True, min_periods=1).mean().values
    return xs, smoothed

def poly_fit(x, y, degree=2):
    coeffs = np.polyfit(x, y, degree)
    y_fit  = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return y_fit, r2, coeffs

def plot_results(scored_df, kaggle_model, kaggle_feature_cols,
                 mimic_model, mimic_feature_cols):

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor('#0f1117')

    ACCENT    = '#00d4aa'
    WARN      = '#ff6b6b'
    MID       = '#ffd93d'
    BG        = '#1a1d27'
    TEXT      = '#e8eaf0'
    GREY      = '#8892a4'
    color_map = {'Low': ACCENT, 'Moderate': MID, 'High': WARN}

    def style_ax(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color=TEXT, fontsize=12, fontweight='bold', pad=12)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2d3142')

    def add_metric_annotation(ax, text):
        ax.text(0.97, 0.04, text, transform=ax.transAxes,
                color=GREY, fontsize=8.5, ha='right', va='bottom',
                bbox=dict(facecolor=BG, edgecolor='#2d3142',
                          boxstyle='round,pad=0.3', alpha=0.85))

    # ─────────────────────────────────────────
    # CHART 1: Mean Fatigue Probability by Hour of Day
    # Uses SEM (not SD) for tight, readable band; y forced [0,1]
    # ─────────────────────────────────────────
    ax1 = axes[0]
    style_ax(ax1, "Mean Fatigue Probability\nby Hour of Day")

    hourly       = scored_df.groupby('Time_of_Day')['fatigue_probability']
    hourly_mean  = hourly.mean()
    hourly_sem   = hourly.sem().fillna(0)
    hourly_count = hourly.count()

    mask  = hourly_count >= MIN_SHIFTS_PER_HOUR
    h_mean = hourly_mean.where(mask).reindex(range(24)).interpolate(limit_direction='both')
    h_sem  = hourly_sem.where(mask).reindex(range(24)).interpolate(limit_direction='both')

    sm_mean = pd.Series(uniform_filter1d(h_mean.fillna(h_mean.mean()).values, size=3),
                        index=h_mean.index)
    sm_sem  = pd.Series(uniform_filter1d(h_sem.fillna(0).values, size=3),
                        index=h_sem.index)

    hours = sm_mean.index.values
    lo    = np.clip(sm_mean.values - sm_sem.values, 0, 1)
    hi    = np.clip(sm_mean.values + sm_sem.values, 0, 1)

    ax1.fill_between(hours, lo, hi, alpha=0.25, color=ACCENT, label='±1 SEM')
    ax1.plot(hours, sm_mean.values, color=ACCENT, linewidth=2.5,
             marker='o', markersize=4, zorder=3)
    ax1.axhline(2/3, color=WARN, linestyle='--', linewidth=1.2,
                alpha=0.8, label='High-risk threshold (0.67)')
    ax1.axvspan(0,  7,  alpha=0.07, color='white', zorder=0, label='Night shift')
    ax1.axvspan(19, 23, alpha=0.07, color='white', zorder=0)

    ax1.set_xlabel('Hour of Day', fontsize=9)
    ax1.set_ylabel('Fatigue Probability', fontsize=9)
    ax1.set_xlim(0, 23)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(range(0, 24, 3))
    ax1.legend(fontsize=8, labelcolor=TEXT, facecolor=BG,
               edgecolor='#2d3142', loc='upper left')

    valid_rows = scored_df.dropna(subset=['Time_of_Day', 'fatigue_probability'])
    rho, pval  = spearmanr(valid_rows['Time_of_Day'], valid_rows['fatigue_probability'])
    print(f"\n  [Chart 1] Spearman ρ (hour vs fatigue prob) = {rho:.3f}  p = {pval:.4f}")
    add_metric_annotation(ax1, f"Spearman ρ = {rho:.3f}  (p = {pval:.3f})")

    # ─────────────────────────────────────────
    # CHART 2: Error Rate/hr vs Decisions/hr by Risk Tier
    # Coloured scatter + LOWESS per tier; Pearson r annotated
    # ─────────────────────────────────────────
    ax2 = axes[1]
    style_ax(ax2, "Error Rate/hr vs Decisions/hr\nby Risk Tier")

    y_cap   = min(scored_df['Error_Rate_per_hour'].quantile(0.95), MAX_ERRORS_PER_HOUR)
    x_cap   = scored_df['Decisions_per_hour'].quantile(0.95)
    plot_df = scored_df[
        (scored_df['Decisions_per_hour'] > 0) &
        (scored_df['Decisions_per_hour'] <= x_cap) &
        (scored_df['Error_Rate_per_hour'] <= y_cap)
    ].copy()

    # Scatter dots coloured by tier
    for tier in ['Low', 'Moderate', 'High']:
        sub = plot_df[plot_df['risk_tier'] == tier]
        ax2.scatter(sub['Decisions_per_hour'], sub['Error_Rate_per_hour'],
                    color=color_map[tier], alpha=0.20, s=14,
                    edgecolors='none', label=f'{tier} (n={len(sub):,})', zorder=2)

    # LOWESS trend per tier
    for tier in ['Low', 'Moderate', 'High']:
        sub = plot_df[plot_df['risk_tier'] == tier]
        if len(sub) >= 15:
            xs, ys = lowess_smooth(sub['Decisions_per_hour'].values,
                                   sub['Error_Rate_per_hour'].values, frac=0.55)
            ax2.plot(xs, ys, color=color_map[tier], linewidth=2.5,
                     alpha=1.0, zorder=4)

    ax2.set_xlabel('Decisions per Hour', fontsize=9)
    ax2.set_ylabel('Error Events per Hour', fontsize=9)
    ax2.set_xlim(0, x_cap)
    ax2.set_ylim(0, y_cap * 1.05)
    ax2.legend(fontsize=7.5, labelcolor=TEXT, facecolor=BG,
               edgecolor='#2d3142', title='Risk Tier',
               title_fontsize=8, markerscale=1.8)

    r_val, r_pval = pearsonr(plot_df['Decisions_per_hour'],
                              plot_df['Error_Rate_per_hour'])
    print(f"  [Chart 2] Pearson r (decisions/hr vs error/hr) = {r_val:.3f}  p = {r_pval:.4f}")
    add_metric_annotation(ax2, f"Pearson r = {r_val:.3f}  (p = {r_pval:.3f})")

    # ─────────────────────────────────────────
    # CHART 3: Fatigue Probability vs Shift Duration
    # Polynomial fit (deg=2); R² annotated; no tier lines
    # ─────────────────────────────────────────
    ax3 = axes[2]
    style_ax(ax3, "Fatigue Probability vs\nShift Duration")

    dur_df = scored_df[
        (scored_df['Hours_Awake'] >= 1.0) &
        (scored_df['Hours_Awake'] <= 12.0)
    ].copy()

    bins       = np.linspace(1, 12, 16)
    bin_labels = 0.5 * (bins[:-1] + bins[1:])
    dur_df['dur_bin'] = pd.cut(dur_df['Hours_Awake'], bins=bins, labels=bin_labels)
    dur_df['dur_bin'] = dur_df['dur_bin'].astype(float)

    bin_stats = dur_df.groupby('dur_bin')['fatigue_probability'].agg(
        ['mean', 'std', 'count']
    ).dropna()
    bin_stats = bin_stats[bin_stats['count'] >= 5]
    bin_stats['sem'] = bin_stats['std'] / np.sqrt(bin_stats['count'])
    bin_stats['ci']  = 1.96 * bin_stats['sem']

    bx  = bin_stats.index.values.astype(float)
    bm  = bin_stats['mean'].values
    bci = bin_stats['ci'].values

    # Polynomial fit
    y_fit, r2, coeffs = poly_fit(bx, bm, degree=2)
    x_smooth = np.linspace(bx.min(), bx.max(), 200)
    y_smooth  = np.polyval(coeffs, x_smooth)

    # CI ribbon
    ax3.fill_between(bx,
                     np.clip(bm - bci, 0, 1),
                     np.clip(bm + bci, 0, 1),
                     alpha=0.25, color=ACCENT, label='95% CI')

    # Bin means as dots
    ax3.scatter(bx, bm, color=ACCENT, s=35, zorder=4, alpha=0.85, label='Bin mean')

    # Polynomial curve
    ax3.plot(x_smooth, np.clip(y_smooth, 0, 1),
             color=WARN, linewidth=2.8, zorder=5, label='Poly fit (deg=2)')

    ax3.axhline(2/3, color=GREY, linestyle='--', linewidth=1.2,
                alpha=0.7, label='High-risk (0.67)')

    ax3.set_xlabel('Hours Active in Shift', fontsize=9)
    ax3.set_ylabel('Fatigue Probability', fontsize=9)
    ax3.set_xlim(1, 12)
    ax3.set_ylim(0, 1)
    ax3.set_xticks(range(1, 13))
    ax3.legend(fontsize=8, labelcolor=TEXT, facecolor=BG, edgecolor='#2d3142')

    print(f"  [Chart 3] Polynomial fit R² (shift duration vs fatigue prob) = {r2:.4f}")
    add_metric_annotation(ax3, f"Poly fit R² = {r2:.4f}")

    # ─────────────────────────────────────────
    fig.suptitle(
        'Decision Fatigue Detection in Clinical Workflows  [v7]\n'
        'Calibrated Within-MIMIC GBM · Fixed Diagnostic Charts',
        color=TEXT, fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'fatigue_analysis_v7.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\n  Chart saved → {out_path}")
    plt.close()


def save_outputs(scored_df):
    out_path = os.path.join(OUTPUT_DIR, 'provider_fatigue_scores_v7.csv')
    cols = [
        'provider_id', 'shift_id', 'shift_start',
        'Hours_Awake', 'Decisions_Made', 'Decisions_per_hour',
        'Task_Switches', 'Task_Switches_per_hour',
        'Avg_Decision_Time_sec', 'Error_Rate', 'Error_Rate_per_hour',
        'Cognitive_Load_Score', 'Time_of_Day',
        'fatigue_probability', 'risk_tier',
        'total_dc_orders', 'total_changes', 'n_orders'
    ]
    scored_df[[c for c in cols if c in scored_df.columns]].to_csv(
        out_path, index=False
    )
    print(f"  Scores saved → {out_path}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Shift windows analyzed  : {len(scored_df):,}")
    print(f"  Unique providers        : {scored_df['provider_id'].nunique():,}")
    for tier in ['Low', 'Moderate', 'High']:
        n   = (scored_df['risk_tier'] == tier).sum()
        pct = n / len(scored_df) * 100
        print(f"  {tier:<10} risk shifts    : {n:,} ({pct:.1f}%)")
    print(f"  Mean decisions/hr       : {scored_df['Decisions_per_hour'].mean():.1f}")
    print(f"  Mean error rate/hr      : {scored_df['Error_Rate_per_hour'].mean():.3f}")
    print(f"\n  Top 5 highest-risk shift windows:")
    top5 = scored_df.nlargest(5, 'fatigue_probability')[[
        'provider_id', 'shift_start', 'fatigue_probability',
        'Error_Rate_per_hour', 'Decisions_per_hour'
    ]]
    print(top5.to_string(index=False))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    kaggle_df                 = load_kaggle_data(KAGGLE_FILE)
    X, y, kaggle_feature_cols = prepare_kaggle_features(kaggle_df)
    kaggle_model, _           = train_kaggle_model(X, y)

    poe_df  = load_poe(POE_FILE)
    emar_df = load_emar(EMAR_FILE) if os.path.exists(EMAR_FILE) else None

    poe_df         = assign_shifts(poe_df)
    poe_df         = compute_error_signals(poe_df, emar_df)
    mimic_features = engineer_mimic_features(poe_df)

    print("\n[3/4] Building within-MIMIC calibrated fatigue model...")
    scored, mimic_model, mimic_feat_cols, scaler = build_within_mimic_model(
        mimic_features,
        kaggle_model.feature_importances_,
        kaggle_feature_cols,
    )

    print("\n[4/4] Generating visualizations and saving results...")
    plot_results(scored, kaggle_model, kaggle_feature_cols,
                 mimic_model, mimic_feat_cols)
    save_outputs(scored)

    print("\n✓ Pipeline complete.")
    print(f"  outputs/fatigue_analysis_v7.png")
    print(f"  outputs/provider_fatigue_scores_v7.csv")