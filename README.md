# Padel With Us — ML Dashboard
## Full-stack: SSMS → Python Flask API → HTML Frontend

---

## FOLDER STRUCTURE
```
padel-dashboard/
├── backend/
│   ├── app.py              ← Flask API (all ML models + SSMS connection)
│   └── requirements.txt
└── frontend/
    └── index.html          ← Dashboard (open directly in browser)
```

---

## STEP 1 — Configure SSMS connection

Open `backend/app.py` and edit the block at line ~30:

```python
SSMS_CONFIG = {
    "server":   "localhost\\SQLEXPRESS",   # ← YOUR server name
    "database": "PadelDW",                 # ← YOUR database name
    "trusted":  True                       # Windows Authentication (recommended)
}
```

**If your table name is different**, change the query on line ~55:
```python
query = "SELECT * FROM fact_performanceF"   # ← adjust table name
```

**SQL Server Authentication** (instead of Windows Auth):
```python
SSMS_CONFIG = {
    "server":   "localhost\\SQLEXPRESS",
    "database": "PadelDW",
    "trusted":  False,
    "user":     "sa",
    "password": "YourPassword"
}
```

---

## STEP 2 — Install ODBC Driver (if not already installed)

Download and install:
- **ODBC Driver 17 for SQL Server**
- https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server

---

## STEP 3 — Install Python dependencies

```bash
cd padel-dashboard/backend
pip install -r requirements.txt
```

**If imbalanced-learn install fails:**
```bash
pip install imbalanced-learn --extra-index-url https://pypi.org/simple
```

---

## STEP 4 — Start the backend

```bash
cd padel-dashboard/backend
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
```

The first request will **train all 4 models** from your live SSMS data.
This takes ~1-3 minutes. After that, results are cached in memory.

---

## STEP 5 — Open the frontend

Simply open `frontend/index.html` in your browser.

**Chrome / Edge:** File → Open File → select index.html  
Or drag index.html into the browser window.

The dashboard automatically connects to `http://localhost:5000`.

---

## WHAT THE DASHBOARD SHOWS

### Overview
- KPI summary cards (TVJ, TEI, TRT, IPJ) live from your data
- Tier distribution bar chart
- High Value Player doughnut chart

### Classification (C section)
- Model comparison table: Accuracy / Precision / Recall / F1 / AUC
- ROC curves for Logistic Regression vs Random Forest
- Confusion matrices (color-coded)
- Feature importance bar chart (RF Classifier)

### Regression (D section)
- Model comparison: R² / RMSE / MAE for Ridge vs RF Regressor
- Actual vs Predicted scatter plots (IPJ score, back-transformed)
- Residual plots (log scale)
- Feature importance (RF Regressor)

### Anomaly Detection
- PCA 2D scatter: normal / IF-only / DBSCAN-only / confirmed anomalies
- Summary cards: anomaly count, PCA explained variance

### KPI Analysis
- Histograms: TVJ, TEI, TRT, RPCO distributions
- Grouped bar: KPI averages by Tier

---

## TROUBLESHOOTING

| Problem | Fix |
|---------|-----|
| `pyodbc.Error: Data source not found` | Install ODBC Driver 17, check server name |
| `ModuleNotFoundError: imblearn` | `pip install imbalanced-learn` |
| Charts empty / CORS error | Make sure backend is running on port 5000 |
| Slow first load | Normal — model training takes 1-3 min on first request |
| `trusted_connection` error | Switch to SQL auth in SSMS_CONFIG |
