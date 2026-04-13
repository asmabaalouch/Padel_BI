"""
Padel With Us — ML Dashboard Backend v4
Covers:
  Section C — Classification  (LR + Random Forest)
  Section D — Regression      (Ridge + RF Regressor)
  Anomaly Detection           (Isolation Forest + DBSCAN on marketing)
  Section E — Clustering      (K-Means + DBSCAN + Hierarchical)
  Section F — Time Series     (Linear trend + XGBoost TS surrogate)
DB: MySQL via SQLAlchemy — mysql+pymysql://root:@localhost:4306/dw_padel
CSV fallback enabled.
"""

import os, traceback, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS

# ── sklearn ──
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold,
    cross_val_score, RandomizedSearchCV
)
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    IsolationForest, GradientBoostingRegressor
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score
)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage as sp_linkage, dendrogram

app = Flask(__name__)
CORS(app)

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
MYSQL_URL = "mysql+pymysql://root:@localhost:4306/dw_padel"
TABLE_NAME = "fact_performance"
CSV_FALLBACK = r"C:\Users\asma\Downloads\fact_performanceF.csv"

FEATURES_CLF = [
    "matchs_joues","victoires","classement_mondial","points",
    "price","real_cost_min","real_cost_max","Nombre_de_spectateurs",
    "scheduled_matches_count","match_reservations_count","Tier_encoded"
]
FEATURES_REG = [
    "matchs_joues","victoires","points","price",
    "real_cost_min","real_cost_max","Nombre_de_spectateurs",
    "scheduled_matches_count","match_reservations_count",
    "Tier_encoded","classement_mondial"
]
FEATURES_CLUSTER = [
    "points","win_rate","points_per_match","classement_mondial",
    "Nombre_de_spectateurs","prize_money_avg","fill_rate",
    "roi","social_score","price","Tier_encoded"
]
CLUSTER_LABELS = {
    0: "Performeurs solides",
    1: "Joueurs en développement",
    2: "Profil intermédiaire",
    3: "Élite mondiale"
}

# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
def load_raw():
    try:
        from sqlalchemy import create_engine
        eng = create_engine(MYSQL_URL)
        df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", con=eng)
        eng.dispose()
        print(f"[DB] {len(df)} rows from MySQL")
        return df, "mysql"
    except Exception as e:
        print(f"[DB] MySQL failed: {e}")
        if CSV_FALLBACK and os.path.exists(CSV_FALLBACK):
            df = pd.read_csv(CSV_FALLBACK)
            print(f"[CSV] {len(df)} rows")
            return df, "csv"
        raise RuntimeError("No data source available.")


def build_features(df):
    """All feature engineering from both notebooks."""
    # Imputation
    for col in ["Nombre_de_spectateurs","match_reservations_count",
                "scheduled_matches_count","likes"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Classification / Regression features
    df["engagement_rate"] = (
        df["likes"] / df["Abonnes_Instagram_Novembre_2025"].replace(0, np.nan)
    ).fillna(0)
    df["prize_avg"] = (df["prize_money_min"] + df["prize_money_max"]) / 2
    commercial = (
        (df["prize_avg"] - df["prize_avg"].min()) /
        (df["prize_avg"].max() - df["prize_avg"].min()) * 100
    )
    df["IPJ_score"] = (
        df["WINS_per_match"]*100*0.40 +
        df["engagement_rate"]*100*0.30 +
        commercial*0.30
    )
    df["high_value_player"] = (df["IPJ_score"] >= df["IPJ_score"].median()).astype(int)
    le = LabelEncoder()
    df["Tier_encoded"] = le.fit_transform(df["Tier"])

    # KPIs
    df["TVJ"]  = (df["victoires"] / df["matchs_joues"]) * 100
    df["TEI"]  = (df["likes"] / df["Abonnes_Instagram_Novembre_2025"]) * 100
    df["TRT"]  = (df["match_reservations_count"] / df["scheduled_matches_count"]) * 100
    df["cout_moyen"]  = (df["real_cost_min"] + df["real_cost_max"]) / 2
    df["prize_moyen"] = (df["prize_money_min"] + df["prize_money_max"]) / 2
    df["RPCO"] = (df["prize_moyen"] / df["cout_moyen"]) * 100
    df["CMMJ"] = df["cout_moyen"] / df["matchs_joues"]

    # Clustering features
    df["win_rate"] = (df["victoires"] / df["matchs_joues"].replace(0, np.nan)).fillna(0)
    df["points_per_match"] = (df["points"] / df["matchs_joues"].replace(0, np.nan)).fillna(0)
    df["fill_rate"] = (
        df["match_reservations_count"] / df["scheduled_matches_count"].replace(0, np.nan)
    ).fillna(0)
    df["prize_money_avg"] = df["prize_avg"]
    df["roi"] = (
        df["prize_money_avg"] / df["cout_moyen"].replace(0, np.nan)
    ).fillna(0)
    df["social_score"] = (
        df["likes"] / df["Abonnes_Instagram_Novembre_2025"].replace(0, np.nan) * 100
    ).fillna(0)

    return df


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────
def train_clf_reg(df):
    X  = df[FEATURES_CLF].copy()
    y  = df["high_value_player"].copy()
    Xr = df[FEATURES_REG].copy()
    yr = np.log1p(df["IPJ_score"].copy())

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=42)

    cv5  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    kf5  = KFold(n_splits=5, shuffle=True, random_state=42)

    # Logistic Regression
    pipe_lr = Pipeline([("sc", StandardScaler()),
                        ("clf", LogisticRegression(random_state=42, max_iter=1000,
                                                    class_weight="balanced",
                                                    solver="liblinear"))])
    s_lr = RandomizedSearchCV(pipe_lr, {"clf__C":[0.01,0.1,1,10,100]},
                              n_iter=5, cv=cv5, scoring="roc_auc",
                              random_state=42, n_jobs=-1)
    s_lr.fit(X_tr, y_tr)

    # Random Forest Classifier
    pipe_rf = Pipeline([("sc", StandardScaler()),
                        ("clf", RandomForestClassifier(random_state=42,
                                                        class_weight="balanced"))])
    s_rf = RandomizedSearchCV(pipe_rf,
        {"clf__n_estimators":[100,200,300],"clf__max_depth":[3,5,7,None],
         "clf__min_samples_split":[2,5,10],"clf__min_samples_leaf":[1,2,4]},
        n_iter=20, cv=cv5, scoring="roc_auc", random_state=42, n_jobs=-1)
    s_rf.fit(X_tr, y_tr)

    # Ridge
    pipe_ridge = Pipeline([("sc", StandardScaler()), ("clf", Ridge())])
    s_ridge = RandomizedSearchCV(pipe_ridge,
        {"clf__alpha":[0.001,0.01,0.1,1,10,50,100,500,1000]},
        n_iter=9, cv=kf5, scoring="r2", random_state=42, n_jobs=-1)
    s_ridge.fit(Xr_tr, yr_tr)

    # Random Forest Regressor
    pipe_rfr = Pipeline([("sc", StandardScaler()),
                         ("clf", RandomForestRegressor(random_state=42))])
    s_rfr = RandomizedSearchCV(pipe_rfr,
        {"clf__n_estimators":[100,200,300],"clf__max_depth":[3,5,7,None],
         "clf__min_samples_split":[2,5,10],"clf__min_samples_leaf":[1,2,4]},
        n_iter=20, cv=kf5, scoring="r2", random_state=42, n_jobs=-1)
    s_rfr.fit(Xr_tr, yr_tr)

    return dict(
        lr=s_lr.best_estimator_, rf=s_rf.best_estimator_,
        ridge=s_ridge.best_estimator_, rfr=s_rfr.best_estimator_,
        X_te=X_te, y_te=y_te, Xr_te=Xr_te, yr_te=yr_te,
        Xr=Xr, yr=yr, kf5=kf5,
    )


def run_anomaly(df):
    feats = ["TEI","likes","Abonnes_Instagram_Novembre_2025"]
    Xa = df[feats].replace([np.inf,-np.inf], np.nan).dropna().copy()
    Xs = StandardScaler().fit_transform(Xa)

    iso   = IsolationForest(n_estimators=200, contamination=0.05,
                             random_state=42, n_jobs=-1)
    if_lb = iso.fit_predict(Xs)
    if_sc = iso.decision_function(Xs)
    db_lb = DBSCAN(eps=0.8, min_samples=5).fit_predict(Xs)

    pca   = PCA(n_components=2, random_state=42)
    Xp    = pca.fit_transform(Xs)
    is_a  = ((if_lb==-1) & (db_lb==-1)).astype(int)

    tmp = Xa.copy()
    tmp["IF_label"]=if_lb; tmp["IF_score"]=if_sc
    tmp["DB_label"]=db_lb; tmp["is_anomaly"]=is_a
    tmp["PCA1"]=Xp[:,0]; tmp["PCA2"]=Xp[:,1]
    tmp["Tier"]=df.loc[Xa.index,"Tier"].values

    anom = tmp[tmp["is_anomaly"]==1]; norm = tmp[tmp["is_anomaly"]==0]
    comparison = []
    for k in feats:
        comparison.append({
            "kpi":k,
            "anomaly": round(float(anom[k].mean()),2) if len(anom) else 0,
            "normal":  round(float(norm[k].mean()),2),
            "ecart":   round(float((anom[k].mean()-norm[k].mean())/norm[k].mean()*100),1) if len(anom) else 0
        })

    cluster_counts = {}
    for lbl in sorted(tmp["DB_label"].unique()):
        name = "Anomalie" if lbl==-1 else f"Cluster {lbl}"
        cluster_counts[name] = int((tmp["DB_label"]==lbl).sum())

    points = [{"x":round(float(tmp.iloc[i]["PCA1"]),3),
               "y":round(float(tmp.iloc[i]["PCA2"]),3),
               "if_label":int(tmp.iloc[i]["IF_label"]),
               "db_label":int(tmp.iloc[i]["DB_label"]),
               "if_score":round(float(tmp.iloc[i]["IF_score"]),4),
               "is_anomaly":int(tmp.iloc[i]["is_anomaly"]),
               "TEI":round(float(tmp.iloc[i]["TEI"]),3),
               "likes":round(float(tmp.iloc[i]["likes"])),
               "followers":round(float(tmp.iloc[i]["Abonnes_Instagram_Novembre_2025"]))}
              for i in range(len(tmp))]

    return {"points":points,"anomaly_count":int(is_a.sum()),
            "normal_count":int((is_a==0).sum()),
            "if_only_count":int(((if_lb==-1)&(db_lb!=-1)).sum()),
            "pca_variance":[round(float(v),3) for v in pca.explained_variance_ratio_],
            "comparison":comparison,"cluster_counts":cluster_counts,
            "tei_normal":[round(float(v),3) for v in norm["TEI"].tolist()],
            "tei_anomaly":[round(float(v),3) for v in anom["TEI"].tolist()],
            "tier_dist":{str(k):int(v) for k,v in anom["Tier"].value_counts().items()}}


def run_clustering(df):
    Xc = df[FEATURES_CLUSTER].copy()
    imp = SimpleImputer(strategy="median")
    Xc_imp = imp.fit_transform(Xc)
    scaler = StandardScaler()
    Xc_sc  = scaler.fit_transform(Xc_imp)

    # Elbow + Silhouette for K selection
    elbow, sils = [], []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km.fit(Xc_sc)
        elbow.append(round(float(km.inertia_)))
        sils.append(round(float(silhouette_score(Xc_sc, km.labels_)), 4))

    K = 4  # interpretable, good balance
    km_final = KMeans(n_clusters=K, init="k-means++", n_init=20, random_state=42)
    km_labels = km_final.fit_predict(Xc_sc)
    df["cluster_kmeans"] = km_labels

    sil_km = round(float(silhouette_score(Xc_sc, km_labels)), 4)
    db_km  = round(float(davies_bouldin_score(Xc_sc, km_labels)), 4)

    # PCA 2D
    pca = PCA(n_components=2, random_state=42)
    Xp  = pca.fit_transform(Xc_sc)
    pca_var = [round(float(v), 3) for v in pca.explained_variance_ratio_]

    centroids_2d = pca.transform(km_final.cluster_centers_)

    # DBSCAN
    db = DBSCAN(eps=1.2, min_samples=5)
    db_labels = db.fit_predict(Xc_sc)
    df["cluster_dbscan"] = db_labels
    n_noise   = int((db_labels==-1).sum())
    n_db_cl   = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    mask_db   = db_labels != -1
    sil_db    = round(float(silhouette_score(Xc_sc[mask_db], db_labels[mask_db])), 4) if n_db_cl>=2 else None
    db_db     = round(float(davies_bouldin_score(Xc_sc[mask_db], db_labels[mask_db])), 4) if n_db_cl>=2 else None

    # Hierarchical — on player aggregates
    pp = df.groupby("ID_player")[FEATURES_CLUSTER].mean()
    Xp_pl = scaler.fit_transform(imp.fit_transform(pp))
    hc = AgglomerativeClustering(n_clusters=min(2, len(pp)), linkage="ward")
    hc_labels = hc.fit_predict(Xp_pl)
    sil_hc = round(float(silhouette_score(Xp_pl, hc_labels)), 4) if len(set(hc_labels))>=2 else None
    db_hc  = round(float(davies_bouldin_score(Xp_pl, hc_labels)), 4) if len(set(hc_labels))>=2 else None

    # Linkage matrix for dendrogram
    Z = sp_linkage(Xp_pl, method="ward")

    # Cluster profiles
    profiles = []
    for c in range(K):
        cdf = df[df["cluster_kmeans"]==c]
        profiles.append({
            "cluster": c,
            "label": CLUSTER_LABELS.get(c, f"Cluster {c}"),
            "n": int(len(cdf)),
            "win_rate": round(float(cdf["win_rate"].mean()), 3),
            "points":   round(float(cdf["points"].mean()), 0),
            "rank":     round(float(cdf["classement_mondial"].mean()), 1),
            "social":   round(float(cdf["social_score"].mean()), 2),
            "tier_top10_pct": round(float((cdf["Tier"]=="Top10").mean()*100), 1),
        })

    # PCA scatter points
    pts = []
    for i in range(len(df)):
        pts.append({"x":round(float(Xp[i,0]),3),"y":round(float(Xp[i,1]),3),
                    "km":int(km_labels[i]),"db":int(db_labels[i]),
                    "player":int(df["ID_player"].iloc[i]),
                    "year":int(df["Year"].iloc[i]),
                    "tier":str(df["Tier"].iloc[i])})

    # Player profiles for hierarchical
    player_pts = []
    for i, pid in enumerate(pp.index):
        player_pts.append({"player":int(pid),"cluster_hc":int(hc_labels[i])})

    # Dendrogram data (merge heights)
    dend_heights = [round(float(v), 3) for v in Z[:, 2]]
    player_names = [f"Joueur {p}" for p in pp.index.tolist()]

    return {
        "elbow": elbow, "silhouettes": sils,
        "K": K, "sil_km": sil_km, "db_km": db_km,
        "pca_variance": pca_var,
        "centroids_2d": [{"x":round(float(c[0]),3),"y":round(float(c[1]),3)} for c in centroids_2d],
        "n_noise": n_noise, "n_db_clusters": n_db_cl,
        "sil_db": sil_db, "db_db": db_db,
        "sil_hc": sil_hc, "db_hc": db_hc,
        "profiles": profiles, "points": pts,
        "player_pts": player_pts, "player_names": player_names,
        "dend_heights": dend_heights,
        "comparison": [
            {"model":"K-Means","silhouette":sil_km,"davies_bouldin":db_km,"n_clusters":K,"outliers":0},
            {"model":"DBSCAN","silhouette":sil_db,"davies_bouldin":db_db,"n_clusters":n_db_cl,"outliers":n_noise},
            {"model":"Hiérarchique","silhouette":sil_hc,"davies_bouldin":db_hc,"n_clusters":2,"outliers":0},
        ]
    }


def run_timeseries(df):
    """
    Time Series using:
    - Linear trend (ARIMA surrogate since statsmodels unavailable)
    - XGBoost/GBR with lag features (XGBoost TS surrogate)
    Data: aggregated by player + year.
    """
    ts = df.groupby(["ID_player","Year"]).agg(
        total_points=("points","sum"),
        avg_win_rate=("win_rate","mean"),
        avg_rank=("classement_mondial","mean"),
        total_prize=("prize_money_avg","sum"),
        avg_spectateurs=("Nombre_de_spectateurs","mean")
    ).reset_index().sort_values(["ID_player","Year"])

    players = sorted(ts["ID_player"].unique().tolist())
    years   = sorted(ts["Year"].unique().tolist())

    # Per-player time series data for charts
    player_series = {}
    for pid in players:
        pdata = ts[ts["ID_player"]==pid].sort_values("Year")
        player_series[str(pid)] = {
            "years": pdata["Year"].tolist(),
            "total_points": [round(float(v)) for v in pdata["total_points"].tolist()],
            "avg_win_rate": [round(float(v),3) for v in pdata["avg_win_rate"].tolist()],
            "avg_rank":     [round(float(v),1) for v in pdata["avg_rank"].tolist()],
        }

    # Stationarity proxy: ADF-like variance ratio
    sta_results = {}
    for pid in players:
        pdata = ts[ts["ID_player"]==pid].sort_values("Year")["total_points"].values
        if len(pdata) >= 3:
            diff1 = np.diff(pdata)
            var_orig = np.var(pdata)
            var_diff = np.var(diff1)
            ratio = var_diff / (var_orig + 1e-9)
            sta_results[str(pid)] = {"var_orig": round(float(var_orig),2),
                                      "var_diff": round(float(var_diff),2),
                                      "stationary": bool(ratio < 1.0)}

    # Linear trend (ARIMA surrogate) — Player 1
    arima_results = {}
    for pid in players:
        pdata = ts[ts["ID_player"]==pid].sort_values("Year")
        X_yr  = pdata["Year"].values.reshape(-1,1)
        y_pts = pdata["total_points"].values
        if len(y_pts) >= 3:
            lr = LinearRegression().fit(X_yr[:-1], y_pts[:-1])
            pred_test = float(lr.predict([[X_yr[-1,0]]])[0])
            f26 = float(lr.predict([[2026]])[0])
            f27 = float(lr.predict([[2027]])[0])
            y_true = float(y_pts[-1])
            mae_v  = abs(y_true - pred_test)
            mape_v = abs((y_true-pred_test)/y_true)*100 if y_true!=0 else 0
            arima_results[str(pid)] = {
                "forecast_2026": max(0, round(f26)),
                "forecast_2027": max(0, round(f27)),
                "mae": round(mae_v), "mape": round(mape_v, 1),
                "r2":  round(float(LinearRegression().fit(X_yr,y_pts).score(X_yr,y_pts)), 3),
                "pred_test": round(pred_test), "true_test": round(y_true),
            }

    # XGBoost TS — lag features, all players
    ts_xgb = ts.copy()
    for lag in [1, 2]:
        ts_xgb[f"pts_lag{lag}"] = ts_xgb.groupby("ID_player")["total_points"].shift(lag)
        ts_xgb[f"wr_lag{lag}"]  = ts_xgb.groupby("ID_player")["avg_win_rate"].shift(lag)

    ts_xgb["year_norm"] = (ts_xgb["Year"] - ts_xgb["Year"].min()) / max(1, ts_xgb["Year"].max()-ts_xgb["Year"].min())
    ts_xgb_clean = ts_xgb.dropna().copy()

    feat_cols = ["ID_player","year_norm","pts_lag1","pts_lag2","avg_win_rate","wr_lag1","avg_rank"]
    feat_cols = [c for c in feat_cols if c in ts_xgb_clean.columns]

    X_xgb  = ts_xgb_clean[feat_cols].values
    y_xgb  = ts_xgb_clean["total_points"].values
    tr_m   = ts_xgb_clean["Year"] < 2025
    te_m   = ts_xgb_clean["Year"] == 2025

    xgb_actual, xgb_pred_vals, xgb_player_ids = [], [], []

    if tr_m.sum() > 0 and te_m.sum() > 0:
        gbr = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                         learning_rate=0.05, random_state=42)
        gbr.fit(X_xgb[tr_m], y_xgb[tr_m])
        y_pred_xgb = gbr.predict(X_xgb[te_m])
        y_true_xgb = y_xgb[te_m]
        mae_xgb  = round(float(mean_absolute_error(y_true_xgb, y_pred_xgb)))
        rmse_xgb = round(float(np.sqrt(mean_squared_error(y_true_xgb, y_pred_xgb))))
        mape_xgb = round(float(np.mean(np.abs((y_true_xgb - y_pred_xgb)/
                                               np.maximum(y_true_xgb, 1)))*100), 1)
        r2_xgb   = round(float(r2_score(y_true_xgb, y_pred_xgb)), 3)
        fi_xgb   = dict(zip(feat_cols, [round(float(v),4) for v in gbr.feature_importances_]))

        for i, idx in enumerate(ts_xgb_clean[te_m].index):
            xgb_actual.append(round(float(y_true_xgb[i])))
            xgb_pred_vals.append(round(float(y_pred_xgb[i])))
            xgb_player_ids.append(int(ts_xgb_clean.loc[idx,"ID_player"]))
    else:
        mae_xgb = rmse_xgb = mape_xgb = r2_xgb = 0
        fi_xgb = {}

    # Model comparison table
    comp_ts = []
    for pid in players:
        if str(pid) in arima_results:
            a = arima_results[str(pid)]
            comp_ts.append({"model":"Linear Trend","player":pid,
                             "mae":a["mae"],"mape":a["mape"],"r2":a["r2"]})
    if mae_xgb > 0:
        comp_ts.append({"model":"XGBoost TS","player":"All",
                        "mae":mae_xgb,"mape":mape_xgb,"r2":r2_xgb})

    # Forecast chart for all players (linear trend)
    forecast_chart = []
    for pid in players:
        pdata = ts[ts["ID_player"]==pid].sort_values("Year")
        forecast_chart.append({
            "player": pid,
            "years":  pdata["Year"].tolist() + [2026, 2027],
            "actual": [round(float(v)) for v in pdata["total_points"].tolist()] + [None, None],
            "forecast": [None]*len(pdata) + [
                arima_results.get(str(pid),{}).get("forecast_2026", 0),
                arima_results.get(str(pid),{}).get("forecast_2027", 0)
            ]
        })

    return {
        "players": players, "years": years,
        "player_series": player_series,
        "stationarity": sta_results,
        "arima_results": arima_results,
        "forecast_chart": forecast_chart,
        "xgb": {"mae":mae_xgb,"rmse":rmse_xgb,"mape":mape_xgb,"r2":r2_xgb,
                "feature_importance":fi_xgb,
                "actual":xgb_actual,"predicted":xgb_pred_vals,"players":xgb_player_ids},
        "comparison": comp_ts,
    }


# ──────────────────────────────────────────────
# CACHE
# ──────────────────────────────────────────────
_cache = {}

def gc():
    if not _cache:
        print("[CACHE] Building…")
        raw, src = load_raw()
        df = build_features(raw)
        m  = train_clf_reg(df)
        _cache.update({"df":df,"source":src,"m":m})
        _cache["anomaly"]   = run_anomaly(df)
        _cache["clustering"]= run_clustering(df)
        _cache["timeseries"]= run_timeseries(df)
        print("[CACHE] Ready.")
    return _cache

def safe(fn):
    try:
        return jsonify(fn())
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error":tb.splitlines()[-1],"trace":tb}), 500


# ──────────────────────────────────────────────
# ENDPOINTS — Overview
# ──────────────────────────────────────────────
@app.route("/api/status")
def status():
    return jsonify({"ok":True})

@app.route("/api/overview")
def overview():
    def _():
        c=gc(); df=c["df"]
        return {
            "source":c["source"],
            "total_records":int(len(df)),
            "total_players":int(df["ID_player"].nunique()),
            "years":sorted(df["Year"].unique().tolist()),
            "avg_TVJ":round(float(df["TVJ"].mean()),1),
            "avg_TEI":round(float(df["TEI"].mean()),3),
            "avg_TRT":round(float(df["TRT"].mean()),1),
            "avg_IPJ":round(float(df["IPJ_score"].mean()),1),
            "high_value_pct":round(float(df["high_value_player"].mean()*100),1),
            "tiers":df["Tier"].value_counts().to_dict(),
            "player_profiles": df.groupby("ID_player")[
                ["points","win_rate","classement_mondial","social_score"]
            ].mean().round(2).reset_index().to_dict(orient="records"),
        }
    return safe(_)

# ──────────────────────────────────────────────
# ENDPOINTS — Classification
# ──────────────────────────────────────────────
@app.route("/api/classification/metrics")
def clf_metrics():
    def _():
        c=gc(); m=c["m"]; out=[]
        for name,mdl in [("Logistic Regression",m["lr"]),("Random Forest",m["rf"])]:
            yp=mdl.predict(m["X_te"]); ypr=mdl.predict_proba(m["X_te"])[:,1]
            out.append({"model":name,
                "accuracy": round(float(accuracy_score(m["y_te"],yp)),3),
                "precision":round(float(precision_score(m["y_te"],yp)),3),
                "recall":   round(float(recall_score(m["y_te"],yp)),3),
                "f1":       round(float(f1_score(m["y_te"],yp)),3),
                "auc":      round(float(roc_auc_score(m["y_te"],ypr)),3)})
        return out
    return safe(_)

@app.route("/api/classification/roc")
def clf_roc():
    def _():
        c=gc(); m=c["m"]; out=[]
        for name,mdl in [("Logistic Regression",m["lr"]),("Random Forest",m["rf"])]:
            ypr=mdl.predict_proba(m["X_te"])[:,1]
            fpr,tpr,_=roc_curve(m["y_te"],ypr)
            out.append({"model":name,"auc":round(float(roc_auc_score(m["y_te"],ypr)),3),
                "fpr":[round(float(x),4) for x in fpr],
                "tpr":[round(float(x),4) for x in tpr]})
        return out
    return safe(_)

@app.route("/api/classification/confusion")
def clf_cm():
    def _():
        c=gc(); m=c["m"]; out=[]
        for name,mdl in [("Logistic Regression",m["lr"]),("Random Forest",m["rf"])]:
            yp=mdl.predict(m["X_te"])
            out.append({"model":name,"matrix":confusion_matrix(m["y_te"],yp).tolist(),
                        "labels":["Standard (0)","High Value (1)"]})
        return out
    return safe(_)

@app.route("/api/classification/feature_importance")
def clf_fi():
    def _():
        c=gc(); m=c["m"]
        imp=pd.Series(m["rf"].named_steps["clf"].feature_importances_,
                      index=FEATURES_CLF).sort_values(ascending=False)
        return {"features":imp.index.tolist(),"values":[round(float(v),4) for v in imp.values]}
    return safe(_)

@app.route("/api/classification/feature_selection")
def clf_fs():
    """Feature selection scores from notebook section A."""
    def _():
        # Pre-computed from notebook markdown cell [16]
        data = [
            {"feature":"points","filter_f":185,"wrapper_rank":1,"embedded_pct":17.9,"verdict":"3/3"},
            {"feature":"classement_mondial","filter_f":208,"wrapper_rank":1,"embedded_pct":9.5,"verdict":"3/3"},
            {"feature":"victoires","filter_f":206,"wrapper_rank":1,"embedded_pct":7.3,"verdict":"3/3"},
            {"feature":"matchs_joues","filter_f":217,"wrapper_rank":4,"embedded_pct":7.3,"verdict":"2/3"},
            {"feature":"match_reservations_count","filter_f":80,"wrapper_rank":1,"embedded_pct":11.4,"verdict":"3/3"},
            {"feature":"real_cost_max","filter_f":0,"wrapper_rank":1,"embedded_pct":17.2,"verdict":"2/3"},
            {"feature":"real_cost_min","filter_f":0,"wrapper_rank":1,"embedded_pct":8.1,"verdict":"2/3"},
            {"feature":"Nombre_de_spectateurs","filter_f":14,"wrapper_rank":1,"embedded_pct":6.2,"verdict":"2/3"},
            {"feature":"price","filter_f":2,"wrapper_rank":1,"embedded_pct":5.1,"verdict":"2/3"},
            {"feature":"Tier_encoded","filter_f":0,"wrapper_rank":4,"embedded_pct":3.2,"verdict":"1/3"},
            {"feature":"scheduled_matches_count","filter_f":1,"wrapper_rank":4,"embedded_pct":2.8,"verdict":"1/3"},
        ]
        return data
    return safe(_)

# ──────────────────────────────────────────────
# ENDPOINTS — Regression
# ──────────────────────────────────────────────
@app.route("/api/regression/metrics")
def reg_metrics():
    def _():
        c=gc(); m=c["m"]; out=[]
        for name,mdl in [("Ridge Regression",m["ridge"]),("Random Forest Regressor",m["rfr"])]:
            yp=mdl.predict(m["Xr_te"])
            cv=cross_val_score(mdl,m["Xr"],m["yr"],cv=m["kf5"],scoring="r2")
            ypo=np.expm1(yp); yto=np.expm1(m["yr_te"].values)
            out.append({"model":name,
                "mse":  round(float(mean_squared_error(m["yr_te"],yp)),5),
                "rmse": round(float(np.sqrt(mean_squared_error(m["yr_te"],yp))),5),
                "mae":  round(float(mean_absolute_error(m["yr_te"],yp)),5),
                "r2":   round(float(r2_score(m["yr_te"],yp)),4),
                "r2_cv_mean":round(float(cv.mean()),4),
                "r2_cv_std": round(float(cv.std()),4),
                "rmse_orig":round(float(np.sqrt(mean_squared_error(yto,ypo))),2),
                "mae_orig": round(float(mean_absolute_error(yto,ypo)),2)})
        return out
    return safe(_)

@app.route("/api/regression/actual_vs_predicted")
def reg_avp():
    def _():
        c=gc(); m=c["m"]; out=[]
        for name,mdl in [("Ridge Regression",m["ridge"]),("Random Forest Regressor",m["rfr"])]:
            yp=np.expm1(mdl.predict(m["Xr_te"]))
            yt=np.expm1(m["yr_te"].values)
            out.append({"model":name,
                "actual":[round(float(v),2) for v in yt],
                "predicted":[round(float(v),2) for v in yp]})
        return out
    return safe(_)

@app.route("/api/regression/residuals")
def reg_res():
    def _():
        c=gc(); m=c["m"]; out=[]
        for name,mdl in [("Ridge Regression",m["ridge"]),("Random Forest Regressor",m["rfr"])]:
            yp=mdl.predict(m["Xr_te"])
            res=m["yr_te"].values-yp
            out.append({"model":name,
                "predicted":[round(float(v),4) for v in yp],
                "residuals":[round(float(v),4) for v in res]})
        return out
    return safe(_)

@app.route("/api/regression/feature_importance")
def reg_fi():
    def _():
        c=gc(); m=c["m"]
        imp=pd.Series(m["rfr"].named_steps["clf"].feature_importances_,
                      index=FEATURES_REG).sort_values(ascending=False)
        return {"features":imp.index.tolist(),"values":[round(float(v),4) for v in imp.values]}
    return safe(_)

# ──────────────────────────────────────────────
# ENDPOINTS — Anomaly
# ──────────────────────────────────────────────
@app.route("/api/anomaly")
def anomaly():
    return safe(lambda: gc()["anomaly"])

# ──────────────────────────────────────────────
# ENDPOINTS — Clustering (Section E)
# ──────────────────────────────────────────────
@app.route("/api/clustering/overview")
def cl_overview():
    return safe(lambda: gc()["clustering"])

@app.route("/api/clustering/profiles")
def cl_profiles():
    return safe(lambda: gc()["clustering"]["profiles"])

@app.route("/api/clustering/comparison")
def cl_comparison():
    return safe(lambda: gc()["clustering"]["comparison"])

# ──────────────────────────────────────────────
# ENDPOINTS — Time Series (Section F)
# ──────────────────────────────────────────────
@app.route("/api/timeseries/overview")
def ts_overview():
    return safe(lambda: gc()["timeseries"])

@app.route("/api/timeseries/forecast")
def ts_forecast():
    return safe(lambda: gc()["timeseries"]["forecast_chart"])

@app.route("/api/timeseries/xgb")
def ts_xgb():
    return safe(lambda: gc()["timeseries"]["xgb"])

@app.route("/api/timeseries/comparison")
def ts_comparison():
    return safe(lambda: gc()["timeseries"]["comparison"])

# ──────────────────────────────────────────────
# ENDPOINTS — KPIs
# ──────────────────────────────────────────────
@app.route("/api/kpi/distribution")
def kpi_dist():
    def _():
        df=gc()["df"]; res={}
        for kpi in ["TVJ","TEI","TRT","RPCO","CMMJ","IPJ_score"]:
            v=df[kpi].replace([np.inf,-np.inf],np.nan).dropna()
            res[kpi]={"values":[round(float(x),2) for x in v.tolist()],
                "mean":round(float(v.mean()),2),"median":round(float(v.median()),2),
                "std":round(float(v.std()),2),"min":round(float(v.min()),2),
                "max":round(float(v.max()),2)}
        return res
    return safe(_)

@app.route("/api/kpi/by_tier")
def kpi_tier():
    def _():
        df=gc()["df"]
        return df.groupby("Tier")[["TVJ","TEI","TRT","RPCO","IPJ_score"]]\
                 .mean().round(2).reset_index().to_dict(orient="records")
    return safe(_)

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
