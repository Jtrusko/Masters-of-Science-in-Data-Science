import pandas as pd, numpy as np, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":0.25,"figure.dpi":200,
                     "axes.spines.top":False,"axes.spines.right":False})
NAVY="#12355B"; AMB="#E8A33D"; RED="#C0392B"; TEAL="#1B9E77"; GREY="#8A8F98"; ICE="#EAF0F6"

W = pd.read_pickle("window_store_labelled.pkl")
FAILS = [("F1","2020-04-18 00:00","2020-04-18 23:59"),("F2","2020-05-29 23:30","2020-05-30 06:00"),
         ("F3","2020-06-05 10:00","2020-06-07 14:30"),("F4","2020-07-15 14:30","2020-07-15 19:00")]

# ---- three-stage signature
pre = pd.Series(False, index=W.index)
for _, s, e in FAILS:
    s = pd.Timestamp(s)
    pre |= (W.index >= s-pd.Timedelta("48h")) & (W.index < s)
stage = pd.Series("healthy", index=W.index)
stage[pre] = "pre-failure (48h)"
stage[W.is_fail==1] = "failure"

sig = W.groupby(stage).agg(
    windows=("duty","size"),
    cycles_per_hour=("cycles_1h","mean"),
    duty=("duty","mean"),
    TP3_min=("TP3_min","mean"),
    oil_T=("Oil_temperature_mean","mean"),
    motor_A=("Motor_current_mean","mean"))
order = ["healthy","pre-failure (48h)","failure"]
sig = sig.loc[order]
print("=== THREE-STAGE DEGRADATION SIGNATURE ===")
print(sig.round(3).to_string())
sig.round(4).to_csv("signature.csv")

hb, pb, fb = sig.cycles_per_hour
print(f"\ncycling rate: healthy {hb:.2f}/h -> pre-failure {pb:.2f}/h ({100*(pb/hb-1):+.0f}%) "
      f"-> failure {fb:.2f}/h ({100*(fb/hb-1):+.0f}%)")

# ---- FIG A: three-stage signature bars
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
metrics = [("cycles_per_hour","cycles per hour"), ("duty","duty fraction"), ("TP3_min","min TP3 (bar)")]
cols = [TEAL, AMB, RED]
for ax,(m,lab) in zip(axes, metrics):
    v = sig[m].values
    ax.bar(range(3), v, color=cols, width=0.62)
    for i,x in enumerate(v):
        ax.text(i, x, f"{x:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(["healthy","pre-48h","failure"], fontsize=8)
    ax.set_title(lab, fontsize=9, loc="left", fontweight="bold")
    ax.set_ylim(0, max(v)*1.28)
axes[2].set_ylim(min(sig.TP3_min)*0.985, max(sig.TP3_min)*1.005)
fig.tight_layout(); fig.savefig("fig/c1_signature.png"); plt.close()

# ---- FIG B: cluster scatter (duty vs cycles), coloured by cluster, failures marked
fig, ax = plt.subplots(figsize=(4.6, 3.1))
n1 = W[W.cluster==1]; n0 = W[W.cluster==0]
ax.scatter(n1.duty, n1.cycles_1h, s=3, c=TEAL, alpha=0.20, lw=0, label="cluster 1: cycling (94.3%)")
ax.scatter(n0.duty, n0.cycles_1h, s=3, c=NAVY, alpha=0.30, lw=0, label="cluster 0: continuous (5.7%)")
f = W[W.is_fail==1]
ax.scatter(f.duty, f.cycles_1h, s=13, facecolors="none", edgecolors=RED, lw=0.7, label="reported failure")
ax.set_xlabel("duty fraction in window"); ax.set_ylabel("load cycles in trailing hour")
ax.legend(frameon=False, fontsize=7.3, loc="upper right")
fig.tight_layout(); fig.savefig("fig/c2_clusters.png"); plt.close()

# ---- FIG C: model comparison
mc = json.load(open("model_comparison.json"))
fig, ax = plt.subplots(figsize=(7.0, 2.9))
names = [m["name"].split("  ",1)[1] if "  " in m["name"] else m["name"] for m in mc]
tags  = [m["name"].split("  ")[0] for m in mc]
f1v = [m["f1"] for m in mc]; apv = [m["pr_auc"] for m in mc]
x = np.arange(len(mc)); w = 0.38
cols_ = [GREY if t.startswith("B") else NAVY for t in tags]
ax.bar(x-w/2, f1v, w, color=cols_, label="F1 (best threshold)")
ax.bar(x+w/2, apv, w, color=AMB, label="PR-AUC")
for i,(a,b) in enumerate(zip(f1v,apv)):
    ax.text(i-w/2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=7)
    ax.text(i+w/2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([f"{t}\n{n}" for t,n in zip(tags,names)], fontsize=6.8)
ax.set_ylabel("score"); ax.set_ylim(0, 0.72)
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
ax.axhline(max(m["f1"] for m in mc if m["name"].startswith("B")), color=RED, ls=":", lw=1)
ax.text(0.02, max(m["f1"] for m in mc if m["name"].startswith("B"))+0.012,
        "best physical baseline", fontsize=7, color=RED)
fig.tight_layout(); fig.savefig("fig/c3_models.png"); plt.close()

# ---- FIG D: lead-time results
lt = json.load(open("leadtime_results.json"))
H = sorted(int(k) for k in lt)
fig, ax = plt.subplots(figsize=(4.6, 2.6))
prev = [lt[str(h)]["prevalence"] for h in H]; ap = [lt[str(h)]["ap"] for h in H]
ax.plot(H, ap, "o-", color=NAVY, lw=1.6, ms=5, label="model PR-AUC (leave-one-event-out)")
ax.plot(H, prev, "s--", color=GREY, lw=1.2, ms=4, label="base rate (random)")
for h,a,p_ in zip(H,ap,prev):
    ax.annotate(f"{a/p_:.1f}x", (h,a), textcoords="offset points", xytext=(0,7),
                ha="center", fontsize=7.5, color=NAVY, fontweight="bold")
ax.set_xlabel("prediction horizon H (hours before onset)")
ax.set_ylabel("PR-AUC"); ax.set_xticks(H); ax.set_ylim(0, 0.26)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig("fig/c4_leadtime.png"); plt.close()

# ---- FIG E: preprocessing funnel
meta = json.load(open("preproc_meta.json"))
fig, ax = plt.subplots(figsize=(7.0, 1.9))
stages = [("raw readings", 1516948), ("on uniform 10s grid", 1841760),
          ("usable after exclusions", 1514623), ("15-min modelling windows", len(W))]
for i,(lab,v) in enumerate(stages):
    x = i*1.85
    ax.add_patch(plt.Rectangle((x,0),1.62,0.92, fc=ICE, ec=NAVY, lw=1.1))
    ax.text(x+0.81, 0.60, f"{v:,}", ha="center", va="center", fontsize=13,
            fontweight="bold", color=NAVY)
    ax.text(x+0.81, 0.26, lab, ha="center", va="center", fontsize=7.2, color="#444")
    if i < 3:
        ax.annotate("", xy=(x+1.82,0.46), xytext=(x+1.64,0.46),
                    arrowprops=dict(arrowstyle="->", color=AMB, lw=1.4))
ax.set_xlim(-0.1, 7.4); ax.set_ylim(-0.05,1.0); ax.axis("off")
fig.tight_layout(); fig.savefig("fig/c5_funnel.png"); plt.close()

print("\nfigures written")
