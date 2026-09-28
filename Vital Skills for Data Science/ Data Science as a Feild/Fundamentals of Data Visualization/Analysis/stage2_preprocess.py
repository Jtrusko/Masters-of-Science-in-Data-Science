"""
Stage 2: Pre-processing pipeline for MetroPT-3.
Every stage is timed so the checkpoint report can quote real throughput.
"""
import pandas as pd, numpy as np, time, json, os

T = {}
def tic(): return time.perf_counter()
def toc(k, t0):
    T[k] = time.perf_counter() - t0
    print(f"  [{k}] {T[k]:.2f}s")

ANA = ["TP2","TP3","H1","DV_pressure","Reservoirs","Oil_temperature","Motor_current"]
DIG = ["COMP","DV_eletric","Towers","MPG","LPS","Pressure_switch","Oil_level","Caudal_impulses"]
SRC = "/mnt/user-data/uploads/MetroPT3_AirCompressor__1_.csv"

print("STAGE 1  load")
t0 = tic()
df = pd.read_csv(SRC, index_col=0, parse_dates=["timestamp"])
toc("load", t0)
n_raw = len(df)

# ---------------------------------------------------------------- STAGE 2
print("STAGE 2  uniform grid + gap flags")
t0 = tic()
df = df.sort_values("timestamp").set_index("timestamp")
dt = df.index.to_series().diff().dt.total_seconds()
GAP_THRESH = 60
gap_mask = (dt > GAP_THRESH)
n_gaps = int(gap_mask.sum())
gap_total_hours = float(dt[gap_mask].sum() / 3600)

# snap to a 10s grid; do NOT interpolate across long gaps
grid = df.resample("10s").mean()
# mark rows that were synthesised
was_present = df.resample("10s").size() > 0
grid["is_synthetic"] = (~was_present).astype(np.int8)
# forward fill only across SHORT holes (<= 6 slots = 60s); leave long gaps NaN
grid[ANA + DIG] = grid[ANA + DIG].ffill(limit=6)
grid["is_gap"] = grid[ANA].isna().any(axis=1).astype(np.int8)
n_grid = len(grid)
n_synth = int(grid.is_synthetic.sum())
n_gap_rows = int(grid.is_gap.sum())
toc("grid", t0)

# ---------------------------------------------------------------- STAGE 3
print("STAGE 3  offset correction (idle-state referenced)")
t0 = tic()
# idle = compressor NOT under load (COMP==1) and motor current near zero
idle = (grid.COMP == 1) & (grid.Motor_current < 0.1)
# Guard: only correct a channel whose idle median is ALREADY near zero. A channel
# that sits at a genuine non-zero pressure when idle (H1 rests near 8.6 bar at the
# cyclonic separator) is measuring something real, not carrying a transducer bias.
# Subtracting its idle median would destroy the signal.
offsets, offset_rejected = {}, {}
for c in ["TP2", "H1", "DV_pressure"]:
    med = float(grid.loc[idle, c].median())
    if abs(med) < 0.5:
        offsets[c] = med
        grid[c] = grid[c] - med
    else:
        offset_rejected[c] = med
toc("offset", t0)

# ---------------------------------------------------------------- STAGE 4
print("STAGE 4  rolling median filter on analogue channels")
t0 = tic()
for c in ANA:
    grid[c] = grid[c].rolling(5, center=True, min_periods=1).median()
toc("median_filter", t0)

# ---------------------------------------------------------------- STAGE 5
print("STAGE 5  digital debounce")
t0 = tic()
MIN_DWELL = 3  # 30 seconds
def debounce(s, min_dwell=MIN_DWELL):
    v = s.to_numpy(copy=True)
    ok = ~np.isnan(v)
    if ok.sum() == 0: return s
    idx = np.flatnonzero(ok)
    vv = v[idx]
    change = np.flatnonzero(np.diff(vv) != 0) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(vv)]))
    for a, b in zip(starts, ends):
        if (b - a) < min_dwell and a > 0:
            vv[a:b] = vv[a-1]
    v[idx] = vv
    return pd.Series(v, index=s.index)

flicker_removed = {}
for c in DIG:
    before = int((grid[c].diff() != 0).sum())
    grid[c] = debounce(grid[c])
    after = int((grid[c].diff() != 0).sum())
    flicker_removed[c] = before - after
toc("debounce", t0)

# ---------------------------------------------------------------- STAGE 6
print("STAGE 6  frozen-channel detection")
t0 = tic()
# Only genuine acquisitions can be "frozen". Rows we forward-filled are identical
# to their predecessor BY CONSTRUCTION, so counting them would measure our own
# imputation rather than a sensor fault.
real = grid.is_synthetic == 0
frozen_step = (grid[ANA].diff().abs().sum(axis=1) == 0) & real
runs = (frozen_step != frozen_step.shift()).cumsum()
run_len = frozen_step.groupby(runs).transform("size")
grid["is_frozen"] = (frozen_step & (run_len >= 6)).astype(np.int8)   # >=60s held
n_frozen = int(grid.is_frozen.sum())
toc("frozen", t0)

grid.to_pickle("clean_grid.pkl")
meta = dict(n_raw=n_raw, n_grid=n_grid, n_synth=n_synth, n_gap_rows=n_gap_rows,
            n_gaps=n_gaps, gap_total_hours=gap_total_hours, n_frozen=n_frozen,
            offsets=offsets, offset_rejected=offset_rejected, flicker_removed=flicker_removed, timings=T)
json.dump(meta, open("preproc_meta.json","w"), indent=2)

print("\n===== PRE-PROCESSING SUMMARY =====")
print(f"raw rows                 {n_raw:,}")
print(f"rows on uniform grid     {n_grid:,}")
print(f"synthesised slots        {n_synth:,} ({100*n_synth/n_grid:.2f}%)")
print(f"rows still NaN (long gap){n_gap_rows:,} ({100*n_gap_rows/n_grid:.2f}%)")
print(f"gaps > 60s               {n_gaps}  totalling {gap_total_hours:.1f} h")
print(f"frozen (>=60s held)      {n_frozen:,} ({100*n_frozen/n_grid:.2f}%)")
print(f"offsets removed          { {k: round(v,4) for k,v in offsets.items()} }")
print(f"offset REJECTED (real)   { {k: round(v,4) for k,v in offset_rejected.items()} }")
print(f"digital flickers removed {flicker_removed}")
print(f"\nTOTAL pre-processing     {sum(T.values()):.2f}s")
print(f"throughput               {n_raw/sum(T.values()):,.0f} readings/s")
