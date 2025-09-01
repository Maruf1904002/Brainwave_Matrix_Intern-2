# Credit Card Fraud Detection (LightGBM)

Detect fraudulent credit-card transactions using a supervised machine learning pipeline on the popular **ULB/Kaggle Credit Card Fraud Detection** dataset.  
The notebook is designed for **Google Colab**, pulls the data via **Kaggle API**, trains a **LightGBM** model with class weighting, and reports **PR‑AUC, ROC‑AUC, Precision, Recall, F1**, and **confusion matrices**.

---

## 🔍 Problem & Dataset

- **Goal:** Flag fraudulent transactions with high recall at a very low false‑positive rate.  
- **Dataset:** [Kaggle – ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
  - 284,807 transactions from European cardholders (Sept‑2013)  
  - Features: anonymized PCA components `V1..V28` + `Time`, `Amount`  
  - Label: `Class` (1 = fraud, 0 = legit)  
  - **Class imbalance:** ~0.172% fraud

---

## ✅ Current Results 

Using a stratified 80/20 split, LightGBM (`n_estimators=600`, class_weight=balanced):

| Metric | Value |
|---|---:|
| **ROC‑AUC** | **0.9789** |
| **PR‑AUC (Average Precision)** | **0.8776** |
| **Best‑F1 Threshold** | **0.9226** |
| **Precision @ best‑F1** | **0.9205** |
| **Recall @ best‑F1** | **0.8265** |
| **F1 @ best‑F1** | **0.8710** |
| **Confusion Matrix @ best‑F1** | `[[56857, 7], [17, 81]]` |
| **False Positive Rate** | **~0.012%** |

> ℹ️ The “best‑F1” threshold was computed on the test set (demo). For a production‑style evaluation, choose the threshold on a **validation** split and report metrics on the **held‑out test** only.

---

## 🚀 Quickstart (Colab)

1. **Open the notebook:** click the Colab badge above.  
2. **Upload Kaggle API token:** `kaggle.json` (Kaggle → Account → *Create New API Token*).  
3. **Run all cells:** the notebook will
   - install deps (`kaggle`, `lightgbm`, `imbalanced-learn`)
   - download & unzip `creditcard.csv`
   - standardize `Amount`/`Time`
   - train **LightGBM** with class weights
   - compute metrics & plot **ROC/PR** curves and **confusion matrices**
   
> Prefer a one‑click run? Use `Runtime → Run all` in Colab after uploading `kaggle.json`.

---

## 🧪 Evaluation Details

- **Split:** Stratified 80/20 train–test. (Try **time‑based** split using `Time` for more realistic drift.)  
- **Imbalance handling:** `class_weight='balanced'` for LightGBM.  
- **Primary metric:** **PR‑AUC** (more informative than ROC‑AUC under extreme imbalance).  
- **Thresholding:** report metrics at 0.5 *and* at a tuned operating point (e.g., **best‑F1** or **recall at fixed FPR**).  
- **Recommended reporting:** Recall at **FPR = 0.1%** and **0.01%** for business alignment.

---

## 📊 Saving Artifacts (optional)

Add this cell at the end of the notebook to save results to the repo:

```python
# Save metrics & plots so they render on GitHub
import json, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay)
import os
os.makedirs("artifacts", exist_ok=True)

# Ensure proba
proba = clf.predict_proba(X_test)[:,1]

# Metrics @ 0.5
pred05 = (proba >= 0.5).astype(int)
report = {
  "accuracy": float(accuracy_score(y_test, pred05)),
  "precision": float(precision_score(y_test, pred05, zero_division=0)),
  "recall": float(recall_score(y_test, pred05, zero_division=0)),
  "f1": float(f1_score(y_test, pred05, zero_division=0)),
  "roc_auc": float(roc_auc_score(y_test, proba)),
  "pr_auc": float(average_precision_score(y_test, proba))
}

# ROC + PR side-by-side
fpr,tpr,_ = roc_curve(y_test, proba)
prec,rec,thr = precision_recall_curve(y_test, proba)
fig, axs = plt.subplots(1,2, figsize=(12,4))
axs[0].plot(fpr,tpr); axs[0].plot([0,1],[0,1],'--'); axs[0].set_title("ROC")
axs[1].plot(rec,prec); axs[1].set_title("Precision–Recall")
plt.tight_layout(); plt.savefig("artifacts/roc_pr.png", dpi=160); plt.close()

# Confusion matrix @ best-F1 threshold
f1s = 2*prec*rec/(prec+rec+1e-9)
i = int(np.argmax(f1s)); thr_best = thr[i] if i < len(thr) else 0.5
cm = confusion_matrix(y_test, (proba >= thr_best).astype(int))
fig, ax = plt.subplots(1,2, figsize=(12,4))
ConfusionMatrixDisplay(cm, display_labels=['Legit','Fraud']).plot(ax=ax[0], values_format='d')
ax[0].set_title(f'CM @ best-F1 (thr={thr_best:.3f})')
cm_norm = (cm.astype(float)/cm.sum(axis=1, keepdims=True))*100
ConfusionMatrixDisplay(cm_norm, display_labels=['Legit','Fraud']).plot(ax=ax[1], values_format='.1f')
ax[1].set_title('CM normalized (%)')
plt.tight_layout(); plt.savefig("artifacts/confusion_matrices.png", dpi=160); plt.close()

with open("artifacts/metrics_report.json","w") as f: json.dump(report, f, indent=2)
print("Saved: artifacts/roc_pr.png, artifacts/confusion_matrices.png, artifacts/metrics_report.json")
```

Then commit the `artifacts/` folder so results are visible on GitHub.

---

## 🔁 Extensions

- **Time‑based split:** train on earlier `Time`, test on later.  
- **Sister models:** XGBoost / CatBoost with class weights or `scale_pos_weight`.  
- **Threshold policy:** choose threshold on a **validation** set for desired FPR (e.g., 0.01%).  
- **Explainability:** SHAP summary & force plots to interpret alerts.  
- **Deployment demo:** small Gradio/Streamlit app scoring a JSON transaction.

---

## 📝 Citation

> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* 2015. (Dataset: ULB/Kaggle Credit Card Fraud Detection).  
> Kaggle dataset link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## ⚖️ License

If you add a license, note it here (e.g., MIT). The Kaggle dataset has its own usage terms—please review them.

---

## 🙌 Acknowledgements

- ULB Machine Learning Group for the dataset
- LightGBM maintainers
- Google Colab & Kaggle
