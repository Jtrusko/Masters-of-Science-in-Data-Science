# MetroPT-3 Checkpoint (Week 4 of 8)

## Deliverables

| File | What it is |
|---|---|
| `MetroPT3_Checkpoint_acmart_overleaf.tex` | Real `\documentclass[sigconf]{acmart}` source. Upload to Overleaf with `fig/`. |
| `MetroPT3_Checkpoint_Report.tex` | Sigconf emulation using `article`; compiles without `acmart.cls`. Produced the PDF. |
| `fig/` | Five figures, all generated from the cleaned data. |
| `analysis/` | The five pipeline scripts plus their JSON/CSV outputs. |

## Reproducing

Run in order. Each writes the inputs the next one needs.

    python analysis/stage2_preprocess.py   # cleaning -> clean_grid.pkl, preproc_meta.json
    python analysis/stage3_windows.py      # windows, clustering, IF -> window_store_labelled.pkl
    python analysis/stage5_diagnose.py     # baseline vs model comparison -> model_comparison.json
    python analysis/stage6_leadtime.py     # early-warning experiment -> leadtime_results.json
    python analysis/stage7_figs.py         # figures + signature.csv

`stage2_preprocess.py` reads the CSV from `/mnt/user-data/uploads/`. Change `SRC`
at the top if your path differs.

Note: the report describes Parquet as the production feature store. This
environment had no `pyarrow` available and no network to install it, so the
scripts write pickle instead. Swap `to_pickle`/`read_pickle` for
`to_parquet`/`read_parquet` where you have pyarrow.

## Headline results

**Pre-processing** (15.6 s total, 97,190 rows/s)
- 909.5 h of the 213-day record missing across 331 gaps: flagged, never interpolated
- Offset correction applied to TP2 and DV_pressure only; H1 rejected by the
  near-zero guard because 8.644 bar at idle is a real pressure, not bias
- Sustained frozen holds: 0.11% of rows (not the 3.35% a naive step-wise count gives)

**Cycle segmentation failed as a modelling unit**
- 10,974 load cycles found; only 3 begin inside a failure window
- Cause: the compressor stops cycling and runs continuously during a leak
- Fix: 15-minute windows as the unit, cycle statistics as features

**Clustering** (k = 2, silhouette 0.676)
- Cluster 0 = continuous-run regime: 5.7% of windows, holds 98.5% of failure windows
- But only 34.0% of cluster 0 is a reported failure: labels are incomplete

**Model comparison** (test = Apr-Sep, 11,885 windows)
| Model | F1 | PR-AUC | Events |
|---|---|---|---|
| B2 instantaneous duty (baseline) | 0.577 | 0.405 | 4/4 |
| M1 Isolation Forest, all 41 feats | 0.312 | 0.136 | 4/4 |
| M2 Isolation Forest, 14 scoped feats | 0.546 | 0.342 | 4/4 |
| M5 scoped IF gated by cluster | 0.580 | 0.372 | 4/4 |

The best learned model beats a one-line threshold by 0.003 F1. Concurrent
detection is solved by physics.

**Early warning** (leave-one-event-out, all in-failure windows removed)
| Horizon | Prevalence | PR-AUC | Lift |
|---|---|---|---|
| 6 h | 0.58% | 0.022 | 3.8x |
| 12 h | 1.17% | 0.045 | 3.8x |
| 24 h | 1.98% | 0.058 | 2.9x |
| 48 h | 4.03% | 0.208 | 5.2x |

**Three-stage signature**
- healthy 2.60 cycles/h -> pre-failure 3.11 (+20%) -> failure 0.13 (-95%)
- duty fraction flat at 0.14 until onset, so duty-based alarms have zero lead time

## Switch rule

If leave-one-event-out PR-AUC lift at the 24 h horizon is still below 3x at the
end of week 6, stop tuning and ship the change-point fallback on the cycle-rate
series.
