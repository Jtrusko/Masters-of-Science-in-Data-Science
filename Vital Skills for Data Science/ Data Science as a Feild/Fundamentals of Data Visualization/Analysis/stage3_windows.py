"""
Stage 3 (revised): cycle segmentation revealed that the compressor STOPS CYCLING
during a failure, so the cycle is not a usable modelling unit inside the events
we most need to model. We therefore keep cycle segmentation as a FEATURE SOURCE
and move the modelling unit to a fixed 15-minute window.
"""
import pandas as pd, numpy as np, time, json, os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             precision_recall_curve, average_precision_score)
from sklearn.ensemble import IsolationForest

T = {}
ANA = ["TP2","TP3","H1","DV_pressure","Oil_temperature","Motor_current"]
DIG = ["COMP","DV_eletric","Towers","MPG","LPS","Pressure_switch","Oil_level","Caudal_impulses"]
FAILS = [("F1","2020-04-18 00:00","2020-04-18 23:59"),
         ("F2","2020-05-29 23:30","2020-05-30 06:00"),
         ("F3","2020-06-05 10:00","2020-06-07 14:30"),
         ("F4","2020-07-15 14:30","2020-07-15 19:00")]

g = pd.read_pickle("clean_grid.pkl")
g = g[(g.is_gap == 0) & (g.is_frozen == 0)].copy()
print(f"usable rows: {len(g):,}")

# ---------------------------------------- CYCLE SEGMENTATION (diagnostic)
print("\n== cycle segmentation diagnostic ==")
load = (g.COMP == 0).astype(np.int8)
edge = load.diff().fillna(0)
starts, ends = g.index[edge == 1], g.index[edge == -1]
if len(ends) and len(starts) and ends[0] < starts[0]: ends = ends[1:]
n = min(len(starts), len(ends)); starts, ends = starts[:n], ends[:n]
cyc = pd.DataFrame({"start": starts, "end": ends})
cyc["dur_s"] = (cyc.end - cyc.start).dt.total_seconds()
cyc = cyc.set_index("start")

failmask = pd.Series(False, index=cyc.index)
for _, s, e in FAILS: failmask |= (cyc.index >= s) & (cyc.index <= e)
print(f"total load cycles: {len(cyc):,}")
print(f"cycles beginning inside a failure window: {int(failmask.sum())}")
print(f"median cycle duration  normal={cyc[~failmask].dur_s.median():.0f}s  "
      f"failure={cyc[failmask].dur_s.median() if failmask.sum() else float('nan'):.0f}s")
print(f"max cycle duration     normal={cyc[~failmask].dur_s.max():.0f}s  "
      f"failure={cyc[failmask].dur_s.max() if failmask.sum() else float('nan'):.0f}s")
print("=> during a failure the unit runs CONTINUOUSLY: few or no cycle boundaries.")

# ---------------------------------------- WINDOW FEATURE STORE
print("\nSTAGE 3  15-minute window feature store")
t0 = time.perf_counter()
R = g.resample("15min")
W = pd.DataFrame(index=R.mean().index)
for c in ANA:
    W[f"{c}_mean"] = R[c].mean(); W[f"{c}_std"] = R[c].std()
    W[f"{c}_min"] = R[c].min();  W[f"{c}_max"] = R[c].max()
W["duty"] = R.apply(lambda x: 0) if False else (g.COMP == 0).resample("15min").mean()
W["n_readings"] = R.size()
# cycles per window, from the segmentation above
cyc_per = pd.Series(1, index=cyc.index).resample("15min").sum().reindex(W.index).fillna(0)
W["n_cycles"] = cyc_per
W["mean_cycle_s"] = cyc.dur_s.resample("15min").mean().reindex(W.index)
W["max_cycle_s"] = cyc.dur_s.resample("15min").max().reindex(W.index)
for c in ["LPS","Oil_level","Caudal_impulses","Towers","Pressure_switch","DV_eletric"]:
    W[f"{c}_frac"] = g[c].resample("15min").mean()
# trailing context: the physics of a leak is cumulative
for w, k in [("1h",4), ("6h",24), ("24h",96)]:
    W[f"duty_{w}"] = W.duty.rolling(k, min_periods=2).mean()
    W[f"cycles_{w}"] = W.n_cycles.rolling(k, min_periods=2).sum()
W["duty_delta_24h"] = W.duty_1h - W.duty_24h
W = W[W.n_readings >= 45]                       # >=50% coverage of the window
W["mean_cycle_s"] = W.mean_cycle_s.fillna(0)
W["max_cycle_s"] = W.max_cycle_s.fillna(0)
W = W.dropna()
T["window_features"] = time.perf_counter() - t0
print(f"  windows: {len(W):,}   features: {W.shape[1]}   [{T['window_features']:.2f}s]")

lab = pd.Series("normal", index=W.index)
for name, s, e in FAILS: lab[(W.index >= s) & (W.index <= e)] = name
W["regime"] = lab; W["is_fail"] = (W.regime != "normal").astype(int)
print(f"  failure windows: {int(W.is_fail.sum())} ({100*W.is_fail.mean():.2f}%)")

print("\n  NORMAL vs FAILURE (window level)")
cmp_ = pd.DataFrame({"normal": W[W.is_fail==0][["duty","n_cycles","max_cycle_s","Oil_temperature_mean","Motor_current_mean","TP3_min"]].mean(),
                     "failure": W[W.is_fail==1][["duty","n_cycles","max_cycle_s","Oil_temperature_mean","Motor_current_mean","TP3_min"]].mean()})
print(cmp_.round(3).to_string())

t0 = time.perf_counter(); W.to_pickle("window_store.pkl"); T["write_store"] = time.perf_counter()-t0
print(f"  store written {os.path.getsize('window_store.pkl')/1e6:.2f} MB [{T['write_store']:.2f}s]")

# ---------------------------------------- CLUSTERING
print("\nSTAGE 4  clustering")
num = W.select_dtypes(include=[np.number]).drop(columns=["is_fail","n_readings"])
num = num.replace([np.inf,-np.inf], np.nan); num = num.fillna(num.median())
sc = StandardScaler().fit(num.values); X = sc.transform(num.values)
p = PCA(n_components=0.95, svd_solver="full").fit(X); Xp = p.transform(X)
print(f"  PCA: {Xp.shape[1]} comps retain 95% var (from {X.shape[1]} features)")

res = []
for k in range(2, 9):
    t0 = time.perf_counter()
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xp)
    el = time.perf_counter()-t0
    sil = silhouette_score(Xp, km.labels_, sample_size=8000, random_state=42)
    db = davies_bouldin_score(Xp, km.labels_)
    res.append((k, sil, db, el)); print(f"  k={k}  sil={sil:.3f}  DB={db:.3f}  ({el:.2f}s)")
best_k = max(res, key=lambda r: r[1])[0]
print(f"  --> k={best_k} selected by silhouette")
km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Xp)
W["cluster"] = km.labels_

prof = W.groupby("cluster").agg(n=("duty","size"), duty=("duty","mean"),
        cycles=("n_cycles","mean"), maxcyc=("max_cycle_s","mean"),
        oilT=("Oil_temperature_mean","mean"), motor=("Motor_current_mean","mean"),
        tp3min=("TP3_min","mean"), fail=("is_fail","mean"))
prof["fail_pct"] = (100*prof.fail).round(2); prof = prof.drop(columns=["fail"])
prof["share_pct"] = (100*prof.n/len(W)).round(2)
print("\n  CLUSTER PROFILE"); print(prof.round(3).to_string())
print("\n  cluster x regime"); print(pd.crosstab(W.cluster, W.regime).to_string())
fc = W[W.is_fail==1].cluster.value_counts(normalize=True)
print(f"\n  {100*fc.iloc[0]:.1f}% of failure windows land in cluster {fc.index[0]}, "
      f"which is {100*prof.loc[fc.index[0],'n']/len(W):.1f}% of all windows")

t0=time.perf_counter(); dbs = DBSCAN(eps=2.0, min_samples=15).fit(Xp[:,:8]); T["dbscan"]=time.perf_counter()-t0
nl = len(set(dbs.labels_))-(1 if -1 in dbs.labels_ else 0)
print(f"\n  DBSCAN: {nl} clusters, {100*(dbs.labels_==-1).mean():.1f}% noise [{T['dbscan']:.2f}s]")
print(f"  of failure windows, {100*(dbs.labels_[W.is_fail.values==1]==-1).mean():.1f}% flagged noise")

# ---------------------------------------- ISOLATION FOREST
print("\nSTAGE 5  Isolation Forest, trained on Feb-Mar only")
tr = W.index < "2020-04-01"
print(f"  train windows: {int(tr.sum()):,}  (failures in train: {int(W[tr].is_fail.sum())})")
t0=time.perf_counter()
iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1).fit(Xp[tr])
T["iso_train"]=time.perf_counter()-t0
t0=time.perf_counter(); s_all = -iso.score_samples(Xp); T["iso_score"]=time.perf_counter()-t0
W["iso"] = s_all
print(f"  [train {T['iso_train']:.2f}s]  [score {len(W):,} windows {T['iso_score']:.2f}s "
      f"= {1000*T['iso_score']/len(W):.3f} ms/window]")

te = W[~tr]
ap = average_precision_score(te.is_fail, te.iso)
prec, rec, thr = precision_recall_curve(te.is_fail, te.iso)
f1 = 2*prec*rec/np.maximum(prec+rec,1e-12); bi = int(np.nanargmax(f1))
print(f"\n  TEST Apr-Sep: {len(te):,} windows, {int(te.is_fail.sum())} positive ({100*te.is_fail.mean():.2f}%)")
print(f"  PR-AUC = {ap:.3f}")
print(f"  best F1 = {f1[bi]:.3f}  (precision {prec[bi]:.3f}, recall {rec[bi]:.3f})")

th = thr[bi]; alarm = te.iso >= th
print("\n  EVENT-LEVEL:")
det = 0; tot = 0
for name, s, e in FAILS:
    if pd.Timestamp(e) < pd.Timestamp("2020-04-01"): continue
    inw = (te.index >= s) & (te.index <= e); tot += 1
    c = bool(alarm[inw].any()); det += c
    print(f"    {name}: {'DETECTED' if c else 'MISSED':8s} {int(alarm[inw].sum())}/{int(inw.sum())} windows")
fp = int((alarm & (te.is_fail==0)).sum()); days = (te.index.max()-te.index.min()).days
print(f"  events detected: {det}/{tot}")
print(f"  false-alarm windows: {fp} over {days} d = {30*fp/days:.1f} per 30 days "
      f"({30*fp/days*0.25:.1f} alarm-hours per 30 d)")

# naive baseline on the same test set for comparison
nb = te.duty_24h
apn = average_precision_score(te.is_fail, nb)
pn, rn, tn = precision_recall_curve(te.is_fail, nb)
f1n = 2*pn*rn/np.maximum(pn+rn,1e-12); bn = int(np.nanargmax(f1n))
print(f"\n  BASELINE (24h duty threshold): PR-AUC={apn:.3f}  best F1={f1n[bn]:.3f} "
      f"(P={pn[bn]:.3f}, R={rn[bn]:.3f})")
print(f"  Isolation Forest lift: F1 {f1n[bn]:.3f} -> {f1[bi]:.3f} "
      f"({100*(f1[bi]-f1n[bn])/f1n[bn]:+.1f}%), PR-AUC {apn:.3f} -> {ap:.3f}")

W.to_pickle("window_store_labelled.pkl")
pd.DataFrame(res, columns=["k","silhouette","davies_bouldin","fit_s"]).to_csv("k_selection.csv", index=False)
json.dump({k: float(v) for k,v in T.items()}, open("stage34_timings.json","w"), indent=2)
json.dump({"best_k":int(best_k),"pr_auc":float(ap),"f1":float(f1[bi]),
           "prec":float(prec[bi]),"rec":float(rec[bi]),
           "base_pr_auc":float(apn),"base_f1":float(f1n[bn]),
           "events_detected":int(det),"events_total":int(tot),
           "fa_per_30d":float(30*fp/days),"n_windows":int(len(W)),
           "n_cycles":int(len(cyc)),"cycles_in_fail":int(failmask.sum())},
          open("results.json","w"), indent=2)
print("\nsaved.")
