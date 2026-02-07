# 🛡️ DriftGuard - Silent Model Drift Detection System

**Team:** Strawhats  
**Problem:** Model Output Drift Causing Silent Degradation in Automated Decision Systems  
**Hackathon:** INNOVEX AI - Round 2  

---

## 🎯 What We Built

DriftGuard detects when AI models silently fail **WITHOUT needing ground truth labels**.

### The Problem
AI models degrade over time but keep making confident predictions:
- Loan approval models become unfair
- Medical diagnoses get worse
- Scholarship decisions become biased
- **Nobody notices because labels come months later or never**

### Our Solution
Monitor model behavior using:
- **KL Divergence** - Distribution changes
- **PSI (Population Stability Index)** - Industry standard
- **Kolmogorov-Smirnov Test** - Statistical drift detection
- **Confidence Calibration** - Model certainty tracking
- **Subpopulation Analysis** - Detect bias against specific groups

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
cd data && python generate_dataset.py

# 3. Run backend
cd ../backend && python api.py

# 4. Open dashboard
# Open frontend/dashboard.html in browser
```

---

## 📊 Demo Scenarios

### Scenario A: Economic Downturn
```
Income distribution shifts from $50k → $40k
DriftGuard detects: PSI > 0.25 (CRITICAL)
```

### Scenario B: Credit Score Degradation
```
Credit scores drop from 680 → 650
DriftGuard detects: KL Divergence spike
```

### Scenario C: Regional Crisis
```
Specific zip codes (90001-90010) see income crash
DriftGuard detects: Subpopulation drift
```

---

## 🏗️ Architecture

```
Production Model → DriftGuard Engine → Alert System → Dashboard
                        ↓
                Rolling Baseline Store
```

---

## 🎓 Judge Feedback Addressed

| Feedback | Implementation |
|----------|----------------|
| ✅ Rolling baseline strategy | 7-day window, updates daily |
| ✅ Confidence proxies | For non-probabilistic models |
| ✅ Low-noise alerts | 3-strike rule before alerting |
| ✅ Subpopulation tracking | Detects bias in specific groups |

---

## 📈 Key Features

- **Real-time monitoring** - Detects drift as it happens
- **Interpretable alerts** - Explains WHAT drifted, BY HOW MUCH
- **Production-ready** - Clean code, tested, documented
- **Scalable** - Handles millions of predictions

---

## 🔬 Technical Details

### Drift Detection Algorithms

**KL Divergence:**
```
D_KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
```

**PSI (Population Stability Index):**
```
PSI = Σ (Actual% - Expected%) * ln(Actual%/Expected%)
```

**K-S Test:**
```
D = max|F_baseline(x) - F_current(x)|
```

### Dataset
- 50,000 loan applications
- Features: age, income, credit_score, debt_ratio, employment_years, zip_code
- 3 controlled drift scenarios

---

## ⚠️ Known Limitations

- Requires minimum 30 samples for reliable detection
- Rolling baseline reset may miss gradual drift
- Subpopulation analysis computationally expensive at scale
- Assumes feature distributions are monitored

---

## 📚 Code Structure

```
driftguard/
├── data/
│   └── generate_dataset.py    # Synthetic data with drift
├── backend/
│   ├── drift_detector.py      # Core algorithms
│   └── api.py                 # FastAPI backend
├── frontend/
│   └── dashboard.html         # Visualization
└── README.md                  # This file
```

---

