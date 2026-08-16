"""
Stage 6: the remaining research question.
Concurrent detection is nearly solved by a single physical feature. The open
problem is EARLY WARNING: predicting an event before onset. This experiment
excludes all windows inside a failure so the model cannot cheat by detecting
the event it is supposed to predict, and validates leave-one-event-out because
there are only four events.
"""
import pandas as pd, numpy as np, json
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

W = pd.read_pickle("window_store_labelled.pkl")
FAILS = [("F1","2020-04-18 00:00"),("F2","2020-05-29 23:30"),
         ("F3","2020-06-05 10:00"),("F4","2020-07-15 14:30")]
ENDS  = {"F1":"2020-04-18 23:59","F2":"2020-05-30 06:00",
         "F3":"2020-06-07 14:30","F4":"2020-07-15 19:00"}

FEATS = ["duty","duty_1h","duty_6h","duty_24h","duty_delta_24h","n_cycles",
         "cycles_1h","cycles_6h","cycles_24h","mean_cycle_s","max_cycle_s",
         "Motor_current_mean","Motor_current_std","Oil_temperature_mean",
         "Oil_temperature_std","TP2_mean","TP2_std","H1_mean","TP3_min","TP3_mean",
         "DV_pressure_mean","LPS_frac","Towers_frac","Oil_level_frac"]

# Exclude windows inside any failure: this is prediction, not detection.
inside = W.is_fail == 1
X_all = W[FEATS].copy()

print("=== EARLY WARNING: predict onset H hours ahead ===")
print("(all windows inside a failure window are removed from both train and test)\n")

results = {}
for H in [6, 12, 24, 48]:
    y = pd.Series(0, index=W.index)
    for name, s in FAILS:
        s = pd.Timestamp(s)
        y[(W.index >= s - pd.Timedelta(hours=H)) & (W.index < s)] = 1
    keep = ~inside
    Xk, yk = X_all[keep], y[keep]
    pos = int(yk.sum())

    # leave-one-event-out: hold out the 30 days surrounding each event in turn
    aps, f1s, dets = [], [], []
    for name, s in FAILS:
        s = pd.Timestamp(s)
        lo, hi = s - pd.Timedelta("15D"), pd.Timestamp(ENDS[name]) + pd.Timedelta("2D")
        test_m = (Xk.index >= lo) & (Xk.index <= hi)
        train_m = ~test_m
        if yk[test_m].sum() == 0 or yk[train_m].sum() < 5: continue
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                           max_depth=4, random_state=42,
                                           class_weight="balanced")
        m.fit(Xk[train_m], yk[train_m])
        p = m.predict_proba(Xk[test_m])[:, 1]
        ap = average_precision_score(yk[test_m], p)
        pr, rc, th = precision_recall_curve(yk[test_m], p)
        f1 = 2*pr*rc/np.maximum(pr+rc, 1e-12)
        b = int(np.nanargmax(f1))
        aps.append(ap); f1s.append(f1[b])
        # did any alarm fire in the pre-window at the best threshold?
        thr = th[b] if b < len(th) else th[-1]
        dets.append(bool((p[yk[test_m].values == 1] >= thr).any()))
    base = yk.mean()
    results[H] = dict(pos=pos, prevalence=float(base), ap=float(np.mean(aps)),
                      f1=float(np.mean(f1s)), lift=float(np.mean(aps)/base),
                      events_flagged=int(sum(dets)), n_folds=len(aps))
    print(f"  H={H:2d}h  pre-windows={pos:4d} ({100*base:.2f}% prevalence)  "
          f"LOEO PR-AUC={np.mean(aps):.3f} (lift {np.mean(aps)/base:.1f}x)  "
          f"F1={np.mean(f1s):.3f}  events flagged {sum(dets)}/{len(dets)}")

print("\n  Interpretation: PR-AUC lift over prevalence is the honest measure here.")
print("  A lift near 1.0 means the model has learned nothing beyond base rate.\n")

# Which features carry the early signal? permutation importance at the best H
bestH = max(results, key=lambda h: results[h]["lift"])
print(f"=== FEATURE SIGNAL AT H={bestH}h ===")
y = pd.Series(0, index=W.index)
for name, s in FAILS:
    s = pd.Timestamp(s)
    y[(W.index >= s - pd.Timedelta(hours=bestH)) & (W.index < s)] = 1
keep = ~inside
Xk, yk = X_all[keep], y[keep]
# simple univariate separation: standardised mean difference
sep = ((Xk[yk == 1].mean() - Xk[yk == 0].mean()) / Xk.std()).sort_values(key=abs, ascending=False)
print(sep.head(10).round(3).to_string())

json.dump(results, open("leadtime_results.json","w"), indent=2)
json.dump({k: float(v) for k, v in sep.head(10).items()}, open("leadtime_sep.json","w"), indent=2)
print("\nsaved leadtime_results.json")
