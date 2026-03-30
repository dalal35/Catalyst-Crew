<div align="center">

<br/>

<pre>
 
 ██████╗ ██████╗  ██████╗ ███╗   ██╗██╗███████╗██╗  ██╗██╗███████╗████████╗
██╔════╝██╔═══██╗██╔════╝ ████╗  ██║██║██╔════╝██║  ██║██║██╔════╝╚══██╔══╝
██║     ██║   ██║██║  ███╗██╔██╗ ██║██║███████╗███████║██║█████╗     ██║   
██║     ██║   ██║██║   ██║██║╚██╗██║██║╚════██║██╔══██║██║██╔══╝     ██║   
╚██████╗╚██████╔╝╚██████╔╝██║ ╚████║██║███████║██║  ██║██║██║        ██║   
 ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝        ╚═╝   

</pre>

# Decision Fatigue Detection in Clinical Workflows

**Detecting degraded physician judgment from behavioral signals alone — before the errors appear.**

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![MIMIC-IV](https://img.shields.io/badge/MIMIC--IV-PhysioNet-00d4aa?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-ff6b6b?style=for-the-badge)

<br/>

*Built for the ISB Integrated Science & Business Challenge — Purdue University, April 2025*

<br/>

</div>

---

## The Problem

> **Judges grant parole 65% of the time at the start of a session. By the end, the rate approaches zero.**
> *(Danziger et al., PNAS 2011)*

Decision fatigue is real, statistically measurable, and invisible to the people experiencing it. In high-stakes professional environments — clinical medicine, aviation, legal adjudication — degraded judgment doesn't announce itself. It just quietly increases the probability of a mistake.

The standard response is to wait for errors and respond to them. We flip that entirely.

**We don't measure fatigue. We measure the behavioral shadow fatigue casts on decisions — and the shadow appears before the errors do.**

---

## What This Does

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   KAGGLE BEHAVIORAL DATASET          MIMIC-IV CLINICAL DATA     │
│   ─────────────────────────          ────────────────────────   │
│   Hours_Awake          ──────────►  Time since shift start      │
│   Decisions_Made       ──────────►  Orders placed this shift    │
│   Avg_Decision_Time    ──────────►  Inter-order timestamp gaps  │
│   Error_Rate           ──────────►  Order amendment rate        │
│   Cognitive_Load_Score ──────────►  Order type entropy          │
│   Task_Switches        ──────────►  Order type transitions      │
│                                                                 │
│        ▼ Train fatigue classifier        ▼ Apply to real data   │
│                                                                 │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  Calibrated GBM  →  Fatigue Probability per Shift Window │  │
│   │                                                          │  │
│   │  🟢 Low (0.00–0.33)   🟡 Moderate (0.33–0.67)           │  │
│   │  🔴 High (0.67–1.00)  →  Break Recommendation Triggered  │  │
│   └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

The pipeline trains on labeled behavioral fatigue data from Kaggle, engineers equivalent features from real MIMIC-IV physician order sequences, and produces a per-shift fatigue risk score for every provider — without any self-reporting.

---

## Key Findings

| Signal | Finding |
|---|---|
| **Hour of Day** | Fatigue probability follows a circadian curve — peaks in early morning hours (0–6h) and mid-evening (18–21h) |
| **Error Rate vs Decisions** | High-risk providers show disproportionate error clustering at moderate decision volumes — not just high volumes |
| **Shift Duration** | Fatigue probability follows a U-shaped polynomial across shift hours — high at onset, dipping mid-shift, rising sharply after hour 9 |
| **Top Predictors** | `Hours_Awake` and `Decisions_Made` dominate Kaggle RF importances; `Error_Rate_per_hour` is the strongest MIMIC signal |

---

## Architecture

```
main.py
│
├── [Step 1]  load_kaggle_data()
│             prepare_kaggle_features()       ← circadian sin/cos encoding
│             train_kaggle_model()            ← RF + LR, AUC reported
│
├── [Step 2]  load_poe()                      ← MIMIC-IV physician orders
│             load_emar()                     ← medication administration records
│             assign_shifts()                 ← 12-hr windows (07:00/19:00 anchored)
│             compute_error_signals()         ← D/C + Change + EMAR not-given
│             engineer_mimic_features()       ← per-shift behavioral features
│
├── [Step 3]  build_within_mimic_model()      ← Kaggle-weighted composite score
│                                                within-provider top-quartile labels
│                                                Calibrated GBM (isotonic, cv=3)
│                                                rank-based uniform probability transform
│
└── [Step 4]  plot_results()                  ← 3-panel diagnostic figure
              save_outputs()                  ← CSV + PNG
```

### Feature Engineering from MIMIC-IV POE

| MIMIC Feature | Source | Kaggle Analog |
|---|---|---|
| `Hours_Awake` | `max(ordertime) - min(ordertime)` per shift | `Hours_Awake` |
| `Decisions_per_hour` | `n_orders / shift_duration` (capped at 60) | `Decisions_Made` |
| `Task_Switches_per_hour` | `order_type` transitions / shift_duration | `Task_Switches` |
| `Avg_Decision_Time_sec` | Median inter-order timestamp gap | `Avg_Decision_Time_sec` |
| `Error_Rate_per_hour` | `(D/C + Change + EMAR not-given) / shift_duration` | `Error_Rate` |
| `Cognitive_Load_Score` | Normalized Shannon entropy of order type mix | `Cognitive_Load_Score` |
| `Time_of_Day_sin/cos` | Circadian encoding of shift start hour | `Time_of_Day` |

### Why Circadian Encoding?

Raw hour-of-day is a discontinuous integer — hour 23 and hour 0 are adjacent in time but numerically far apart. Sine/cosine encoding wraps the 24-hour cycle into a continuous circular representation that any linear model can reason about correctly.

---

## Data

### Training Data
**[Human Decision Fatigue Behavioral Dataset](https://www.kaggle.com/datasets/sonalshinde123/human-decision-fatigue-behavioral-dataset)**
— Kaggle. Labeled behavioral fatigue signals with ground truth `Fatigue_Level`.

### Application Data
**[MIMIC-IV](https://physionet.org/content/mimiciv/)**
— PhysioNet / MIT Laboratory for Computational Physiology. Free access with credentialing (complete the CITI training course at physionet.org — approximately 2 hours).

Files used:

```
data/
  decision_fatigue_dataset.csv    ← Kaggle training data
  poe.csv                         ← MIMIC-IV: physician order entry
  emar.csv                        ← MIMIC-IV: medication administration record
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/your-org/catalyst-crew.git
cd catalyst-crew

# 2. Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn scipy matplotlib

# 3. Place data files in data/
#    (see Data section above for sources)

# 4. Run
python3 main.py
```

### Outputs

```
outputs/
  fatigue_analysis_v7.png          ← 3-panel diagnostic figure
  provider_fatigue_scores_v7.csv   ← per-shift fatigue scores for every provider
```

---

## Model Details

### Stage 1 — Kaggle RF Classifier
Trained on labeled behavioral data. Used primarily to extract **feature importances**, which seed the composite score weights applied in Stage 2.

- Algorithm: `RandomForestClassifier` (200 trees, max_depth=6, balanced class weight)
- Validation: AUC on 20% holdout + 5-fold CV
- Features: 8 behavioral signals (circadian-encoded)

### Stage 2 — Within-MIMIC Calibrated GBM
Because MIMIC has no ground-truth fatigue labels, we generate pseudo-labels within each provider using their own shift distribution (top quartile = fatigued). This controls for between-provider baseline differences — a fast physician who slows down is more fatigued than a naturally slow one at the same speed.

- Algorithm: `GradientBoostingClassifier` wrapped in `CalibratedClassifierCV` (isotonic, cv=3)
- Labeling: within-provider top-quartile composite score
- Output transform: rank-based uniform with jitter (guarantees calibrated probability spread)
- Risk tiers: equal thirds — Low [0, 0.33), Moderate [0.33, 0.67), High [0.67, 1.0]

### Why Personal Baseline Matters

Aggregate models that compare all physicians to a population mean conflate two entirely different things:
- **Inter-provider variance** — some physicians are naturally faster or slower
- **Intra-provider fatigue signal** — the same physician degrading over a shift

We model only the second. This is the same methodological insight behind the Israeli parole study — the *change within* a session, not the *level across* sessions, is the fatigue signal.

---

## Diagnostic Charts

### Chart 1 — Fatigue Probability by Hour of Day
Mean fatigue probability (±1 SEM) across all shift windows, grouped by shift start hour. Night-shift windows shaded. **Metric: Spearman ρ** between hour of day and fatigue probability.

### Chart 2 — Error Rate/hr vs Decisions/hr by Risk Tier
Scatter plot coloured by risk tier with LOWESS trend lines per tier. Shows that high-risk providers have disproportionately elevated error rates across the full decision volume range — not just at extremes. **Metric: Pearson r** between decisions/hr and error rate/hr.

### Chart 3 — Fatigue Probability vs Shift Duration
Binned mean fatigue probability across shift hours 1–12, with 95% CI ribbon and degree-2 polynomial fit. Captures the characteristic U-shape: elevated at shift onset, dipping mid-shift, rising sharply in the final hours. **Metric: Polynomial fit R².**

---

## The Business Case

> One missed tumor that leads to a lawsuit costs a radiology group $2–4M.
> A wrongful parole decision costs a jurisdiction far more.
> This tool priced at $50K/year per facility looks like insurance, not software.

**The product is not a wellness tool. It is a liability reduction instrument.**

The model does one thing: sends a **break recommendation** to a supervisor when a provider's fatigue score crosses a threshold. It does not override clinical decisions. It does not surveil employees. It gives the institution a defensible, data-grounded intervention trigger that replaces "we had no way of knowing."

**Target markets:**
- Hospital systems (ICU physician scheduling)
- Radiology groups (reading session management)
- Air traffic control facilities (FAA duty period compliance)
- Legal operations (document review session limits)

---

## Team

Built by **Catalyst Crew** for the ISB Integrated Science & Business Challenge, Purdue University.

| | Role |
|---|---|
| **Yash Dalal** | Data Science & Statistics — statistical modeling, feature engineering, MIMIC pipeline |
| **Sri Dhruti Sirigibathina** | Integrated Business & Engineering — agentic systems, supply chain application, business model |

---

## References

- Danziger, S., Levav, J., & Avnaim-Pesso, L. (2011). *Extraneous factors in judicial decisions.* PNAS.
- Johnson, A.E.W. et al. (2023). *MIMIC-IV, a freely accessible electronic health record dataset.* Scientific Data.
- Drew, T., Vo, M.L.H., & Wolfe, J.M. (2013). *The invisible gorilla strikes again.* Psychological Science.

---

<div align="center">

<br/>

*"We don't measure fatigue. We measure the behavioral shadow fatigue casts on decisions — and the shadow appears before the errors do."*

<br/>

![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![MIMIC-IV](https://img.shields.io/badge/Data-MIMIC--IV-00d4aa?style=flat-square)
![Purdue](https://img.shields.io/badge/Purdue-ISB%20Challenge%202025-CEB888?style=flat-square)

</div>
