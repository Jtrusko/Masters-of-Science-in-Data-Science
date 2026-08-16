"""
Stage 5b: diagnose why the general-purpose anomaly detector loses to a
one-feature physical baseline, and test two targeted fixes.
"""
import pandas as pd, numpy as np, json, time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, average_precision_score

W = pd.read_pickle("window_store_labelled.pkl")
FAILS = [("F1","2020-04-18 00:00","2020-04-18 23:59"),
         ("F2","2020-05-29 23:30","2020-05-30 06:00"),
         ("F3","2020-06-05 10:00","2020-06-07 14:30"),
         ("F4","2020-07-15 14:30","2020-07-15 19:00")]

def evaluate(name, score, index, y):
    ap = average_precision_score(y, score)
    p, r, t = precision_recall_curve(y, score)
    f1 = 2*p*r/np.maximum(p+r, 1e-12); b = int(np.nanargmax(f1))
    th = t[b] if b < len(t) else t[-1]
    alarm = pd.Series(score >= th, index=index)
    det = 0; tot = 0
    for n_, s, e in FAILS:
        if pd.Timestamp(e) < pd.Timestamp("2020-04-01"): continue
        inw = (index >= s) & (index <= e); tot += 1
        det += bool(alarm[inw].any())
    fp = int((alarm.values & (y == 0)).sum())
    days = (index.max()-index.min()).days
    print(f"  {name:38s} PR-AUC={ap:.3f}  F1={f1[b]:.3f}  P={p[b]:.3f}  R={r[b]:.3f}  "
          f"events={det}/{tot}  FA/30d={30*fp/days:6.1f}")
    return dict(name=name, pr_auc=float(ap), f1=float(f1[b]), prec=float(p[b]),
                rec=float(r[b]), events=det, events_tot=tot, fa30=float(30*fp/days))

tr = W.index < "2020-04-01"
te_idx = W.index[~tr]; y = W[~tr].is_fail.values
out = []

print("=== BASELINES AND DIAGNOSIS (test = Apr-Sep) ===\n")
out.append(evaluate("B1  24h duty threshold", W[~tr].duty_24h.values, te_idx, y))
out.append(evaluate("B2  instantaneous duty", W[~tr].duty.values, te_idx, y))
out.append(evaluate("B3  cycles per 6h (inverted)", -W[~tr].cycles_6h.values, te_idx, y))

# --- full-feature IF (what we ran before)
num_all = W.select_dtypes(include=[np.number]).drop(columns=["is_fail","n_readings","cluster","iso"], errors="ignore")
num_all = num_all.replace([np.inf,-np.inf], np.nan); num_all = num_all.fillna(num_all.median())
Xa = StandardScaler().fit_transform(num_all.values)
Pa = PCA(n_components=0.95, svd_solver="full").fit(Xa); Xap = Pa.transform(Xa)
iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1).fit(Xap[tr])
out.append(evaluate(f"M1  IF, all {Xa.shape[1]} feats via PCA", -iso.score_samples(Xap[~tr]), te_idx, y))

# --- DIAGNOSIS: is the problem drift, or feature dilution?
print("\n  DIAGNOSIS")
print(f"  mean duty  Feb-Mar (train) = {W[tr].duty.mean():.3f}")
print(f"  mean duty  Apr-Sep (test)  = {W[~tr].duty.mean():.3f}  "
      f"({100*(W[~tr].duty.mean()/W[tr].duty.mean()-1):+.0f}% vs train)")
sc_tr = -iso.score_samples(Xap[tr]); sc_te = -iso.score_samples(Xap[~tr])
print(f"  IF score   train mean={sc_tr.mean():.3f}   test mean={sc_te.mean():.3f}  "
      f"test-normal mean={sc_te[y==0].mean():.3f}  test-failure mean={sc_te[y==1].mean():.3f}")
print("  => the detector rates ordinary Apr-Sep operation as anomalous relative to")
print("     the Feb-Mar reference, so its alarm budget is spent on seasonal drift.")

# --- FIX 1: physics-scoped feature subset
PHYS = ["duty","duty_1h","duty_6h","duty_24h","duty_delta_24h","n_cycles",
        "cycles_1h","cycles_6h","Motor_current_mean","Oil_temperature_mean",
        "TP2_mean","H1_mean","DV_pressure_mean","LPS_frac"]
Xs = StandardScaler().fit_transform(W[PHYS].values)
iso2 = IsolationForest(n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1).fit(Xs[tr])
out.append(evaluate(f"M2  IF, {len(PHYS)} physics-scoped feats", -iso2.score_samples(Xs[~tr]), te_idx, y))

# --- FIX 2: rolling reference (retrain on trailing 30 days), physics-scoped
print()
t0 = time.perf_counter()
scores = np.zeros(len(W)); scores[:] = np.nan
idx = W.index
step = 96*7                     # refit weekly
first_test = int(np.argmax(~tr))
for start in range(first_test, len(W), step):
    stop = min(start+step, len(W))
    ref_lo = idx[start] - pd.Timedelta("30D")
    ref = (idx >= ref_lo) & (idx < idx[start])
    if ref.sum() < 500: ref = tr
    m = IsolationForest(n_estimators=100, contamination=0.02, random_state=42, n_jobs=-1).fit(Xs[ref])
    scores[start:stop] = -m.score_samples(Xs[start:stop])
roll_time = time.perf_counter()-t0
out.append(evaluate("M3  IF, rolling 30d ref + scoped", scores[~tr], te_idx, y))
print(f"      (rolling refit total {roll_time:.1f}s for {len(range(first_test,len(W),step))} refits)")

# --- FIX 3: cluster-membership detector (from k=2 clustering)
out.append(evaluate("M4  cluster-0 membership", (W[~tr].cluster==0).astype(float).values, te_idx, y))

# --- combined: scoped IF score restricted to the continuous-run regime
comb = np.where(W[~tr].cluster.values==0, -iso2.score_samples(Xs[~tr]), -1e6)
out.append(evaluate("M5  M2 gated by cluster-0", comb, te_idx, y))

json.dump(out, open("model_comparison.json","w"), indent=2)
best = max(out, key=lambda d: d["f1"])
print(f"\n  BEST BY F1: {best['name']}  (F1={best['f1']:.3f})")
