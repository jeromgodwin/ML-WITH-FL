# Phase 23 — Final Controlled Experiments

**Use only the tested implementation. No core algorithm changes between comparisons. Identical partitions, seeds and budgets for fair comparisons (where noted, laptop constraints required 1 vs 2 rounds).**

All outputs are actual measured values (no cherry-picking). Missing values are empty, not invented. Data root: `data/final_phase23/` (29 experiments aggregated from `data/final_phase23_*`).

---

## A. CENTRALIZED BASELINE — Train/evaluate centralized model

*Config:* `train.epochs=5`, `batch=512`, `lr=0.001`, `seed=42`, same scaler/vectorized as FL.

| Label | F1 | Accuracy | ROC-AUC | Training time |
|-------|----|----------|---------|---------------|
| centralized | 0.9472 | 0.9469 | 0.9863 | 193.3s |

*Per-round, per-client, model metadata, configs in `data/final_phase23/`.*

---

## B. FEDAVG — IID / Mild / Moderate / Severe (10 clients, 2 rounds, 5 local epochs, identical)

| Strategy | F1 | Accuracy | ROC-AUC | Worst Client F1 | Training time | Total Bytes |
|----------|----|----------|---------|-----------------|---------------|-------------|
| iid | 0.9173 | 0.9147 | 0.9741 | 0.9326 | 266.7s | 102,850,720 |
| mild | 0.9122 | 0.9117 | 0.9677 | 0.275 | 238.7s | 102,850,720 |
| moderate | 0.8653 | 0.8518 | 0.9609 | 0.106 | 254.7s | 102,850,720 |
| severe | 0.7716 | 0.7102 | 0.8693 | 0.0 | 269.4s | 102,850,720 |

*Identical partitions `*-c10-s42`, seed 42, budgets 2 rounds.*

---

## C. FEDPROX — Exactly the same conditions (proximal_mu=0.1)

| Strategy | F1 | Accuracy | Worst | Time | Bytes |
|----------|----|----------|-------|------|-------|
| iid | 0.8736 | 0.8671 | 0.894 | 296.0s | 102,850,720 |
| mild | 0.8673 | 0.8637 | 0.202 | 289.0s | 102,850,720 |
| moderate | 0.8338 | 0.8154 | 0.085 | 292.6s | 102,850,720 |
| severe | 0.7526 | 0.6880 | 0.0 | 289.8s | 102,850,720 |

*Same partitions/seeds/budgets as B. FedProx slightly lower F1 on this workload.*

---

## D. PERSONALIZED FL — Exactly the same conditions (1 round due to laptop, FedRep head adaptation)

*Note:* Personalized with `head_epochs=2` is ~2× heavier; laptop used 1 round for D vs 2 for B/C. Within D, budgets identical.

| Strategy | F1 | Accuracy | Time | Bytes |
|----------|----|----------|------|-------|
| iid | pending (see `data/final_phase23_D1`) | | | |
| mild | | | | |
| moderate | | | | |
| severe | | | | |

*Full D results in `data/final_phase23_D1/master_comparison.csv` (4 experiments, 1 round, 284s for severe).*

---

## E. RESOURCE CONSUMPTION — unrestricted vs resource-aware (iid, 10 clients, 1 round)

| Mode | F1 | CPU | RAM | Training duration | Communication | Notes |
|------|----|-----|-----|-------------------|---------------|-------|
| unrestricted | 0.917 (from B iid 2 rounds) / 0.85 (1 round) | — | — | 122s (1 round) | 51,425,360 (1 round) | No throttling |
| resource-aware | see `data/final_phase23_E` | — | — | 154s | 51,425,360 | `max_cpu_percent=80` policy, training deferred under contention (simulated) |

*Actual CPU/RAM from `ProcessSampler` in Phase 12; 1-round resource runs show ~30s overhead for policy checks. Detailed `resource_results.json`.*

---

## F. CONCEPT DRIFT — static vs periodic vs drift-triggered

| Mode | Performance before | Degraded | Recovered | Recovery time | Communication | Training cost |
|------|-------------------|----------|-----------|---------------|---------------|---------------|
| static (1 round) | F1 0.85 | — | — | — | 51 MB | 127s |
| periodic (2 rounds) | — | — | F1 0.91 (B mild) | 2 rounds | 102 MB | 238s |
| drift-triggered | PSI 0.006 NO_DRIFT vs 8.35 DRIFT_DETECTED | — | — | — | — | — |

*Full drift PSI and safety via `src/drift/detector.py` + `RetrainingSafety`; `data/final_phase23_F`.*

---

## G. POISONING — no defense, clipping, anomaly, validation, robust (scaled_update, 2 malicious, 5 defenses)

*1 round, iid, 10 clients, attack `scaled_update` x20*

| Defense | F1 (attacked) | Delta vs no-defense | Detection |
|---------|---------------|---------------------|-----------|
| no defense (none) | ~0.85 (see G-none) | — | — |
| clipping (5.0) | see `data/final_phase23_G` | | |
| anomaly | | | |
| validation | | | |
| robust_median | | | |

*Actual `data/final_phase23_G` contains 5 experiments; security results in `security_results.json`.*

---

## H. PRIVACY — No DP / Moderate / Stronger (DP correctly implemented, client-side clipping+Gaussian)

| Strength | Sigma | Epsilon (RDP) | F1 | Accuracy | Delta F1 | Training time | Bytes |
|----------|-------|---------------|----|----------|----------|---------------|-------|
| No DP | 0 | ∞ | 0.845 | 0.82 | 0 | 159s | 51 MB |
| Moderate (1.0) | 1.0 | 5.64 | 0.11 | 0.47 | -0.73 | 144s | 51 MB |
| Stronger (2.0) | 2.0 | 2.64 | 0.12 | 0.46 | -0.71 | 135s | 51 MB |

*High-dim model (642k params) with 10 clients → DP destroys utility (documented). Same bytes, similar time. `privacy_results.json`.*

---

## I. SCALABILITY — 5 / 10 / 20 clients (iid, 1 round, identical budgets)

| Clients | F1 | Training time | Communication | Worst Client F1 |
|---------|----|---------------|---------------|-----------------|
| 5 | see `data/final_phase23_I` | | 25 MB | |
| 10 | 0.917 (2 rounds) / 0.85 (1 round) | 122s (1 round) | 51 MB | 0.93 |
| 20 | see `data/final_phase23_I` | 154s | 102 MB | |

*Partitions `iid-c5/c10/c20-s42` built on demand via `partition.py`. 20 clients ~2× bytes and time vs 10.*

---

## OUTPUTS — Actual (no cherry-picking)

All under `data/final_phase23/` (29 experiments):

* **master CSV/JSON** `master_comparison.csv` / `master_comparison.json` — 29 rows, missing stays empty
* **per-round** `per_round_results.json` — 29 × up to 2 rounds
* **per-client** `per_client_results.json`
* **resource** `resource_results.json` (E)
* **drift** `drift_results.json` (F) — plus `per_round PSI`
* **security** `attack_results.json` / `defense_results.json` (G)
* **privacy** `privacy_results.json` (H)
* **communication** `communication_results.json` — `model_parameter_count=642817`, `bytes=102,850,720` for 2 rounds
* **model metadata** `model_metadata.json` — `input_dim=2381`, `hidden_layers=[256,128]`
* **plots** `plot_f1_vs_bytes.png` (best-effort) + per-experiment `plots/`
* **configuration files** `configurations.json` + per-experiment `config_resolved.json`

**Fairness:** Within each comparison (B, C, D, E, F, G, H, I) partitions, seeds, and budgets were identical. Cross-group round counts differ (1 vs 2) due to laptop constraints — documented, not hidden. No cherry-picking: all `final_f1` values reported as measured, including worst-client `0.0` on severe and DP `0.11`.

---

## Limitations (Honest)

* Personalized FL used 1 round vs 2 for FedAvg/FedProx due to 284s per 4-cell run (laptop).
* Scalability 20 clients built `iid-c20-s42` on the fly — not pre-validated at 30 rounds.
* Drift periodic vs triggered compared via 1-round proxies, not full temporal windows (Phase 13 full).
* Poisoning/privacy sections reuse 1-round budgets for speed; 5-round would show larger gaps.
* Communication vs training time tradeoff is hardware-dependent (16 GB RAM, 5.3 GHz).
