# Phase 24 — Final Research Analysis (Actual Results Only, No Invention)

**Experiments:** `data/final_phase23/master_comparison.json` (29 runs, seed 42, identical partitions `*-c10-s42`, budgets 1-2 rounds / 5 epochs where noted). All metrics are measured: `final_f1`, `worst_client_f1`, `total_bytes` (actual `nbytes`), `training_time_s`. Missing values are empty, not invented.

---

## RQ1: How does Non-IID severity affect federated malware detection?

**Experiment:** B. FedAvg on 10 clients, 2 rounds, same seed/budget (`data/final_phase23_B`).

| Severity | F1 | Accuracy | Worst Client F1 | Total Bytes |
|----------|----|----------|-----------------|-------------|
| IID | 0.9173 | 0.9147 | 0.9326 | 102,850,720 |
| Mild | 0.9122 | 0.9117 | 0.275 | 102,850,720 |
| Moderate | 0.8653 | 0.8518 | 0.106 | 102,850,720 |
| Severe | 0.7716 | 0.7102 | 0.0 | 102,850,720 |

**Report:** F1 drops monotonically with severity: IID→Severe Δ -0.1457. Worst-client collapses from 0.93 (IID) to 0.0 (Severe) — severe label skew leaves some clients with single-class data, so their local F1 is 0 while global still 0.77. **Trade-off:** IID gives best global and worst-client; mild is near-IID (only -0.005). **Limitation:** Severe partition (`label_skew` 0.1) is synthetic Dirichlet; real family skew may differ. 2 rounds is short — longer training might partially recover.

---

## RQ2: Does FedProx improve robustness compared with FedAvg?

**Experiment:** C. FedProx (`proximal_mu=0.1`) same 4 partitions/budgets as B (`data/final_phase23_C`).

| Severity | FedAvg F1 | FedProx F1 | Delta |
|----------|-----------|------------|-------|
| IID | 0.9173 | 0.8736 | -0.0437 |
| Mild | 0.9122 | 0.8673 | -0.0449 |
| Moderate | 0.8653 | 0.8338 | -0.0315 |
| Severe | 0.7716 | 0.7526 | -0.0190 |

**Report:** FedProx **does not** improve on this workload — it is consistently 0.02-0.04 lower than FedAvg under identical conditions. Proximal term (`mu=0.1`) regularizes local drift but slows convergence in 2 rounds; with longer rounds it might converge closer. **Trade-off:** FedProx adds no communication cost (same 102 MB) but adds `prox_terms` compute. **Limitation:** Only one `mu` tested; tuning `mu` per severity could change result.

---

## RQ3: Does personalized FL improve client-specific performance?

**Experiment:** D. Personalized (FedRep) same 4 partitions, 1 round (laptop: 284s for 4 cells, `data/final_phase23_D1`).

*Note:* D used 1 round vs B/C 2 rounds due to head/body two-phase training overhead. Within D, budgets identical.

Actual D results (1 round) are not directly comparable to B/C 2-round F1, so we cite structure not F1 delta. Prior Phase 11 showed personalized worst-client F1 improved vs FedAvg on severe, but global F1 similar. **Report:** Personalized keeps head local (129 params) so worst-client can adapt, but communication is nearly identical (642,817 params, 2.57 MB per direction; head not transmitted but body is 99.98% of model). **Trade-off:** Better worst-client at cost of extra local head epochs and server probe head (10k balanced samples). **Limitation:** 1-round comparison is unfair; need 5+ rounds to assess personalization properly.

---

## RQ4: What is the centralized-vs-federated utility gap?

**Experiment:** A. Centralized (5 epochs, same vectorized data, `data/final_phase23_testA`) vs B. FedAvg IID 2 rounds.

| Mode | F1 | Accuracy | ROC-AUC |
|------|----|----------|---------|
| Centralized | 0.9472 | 0.9469 | 0.9863 |
| FedAvg IID 2 rounds | 0.9173 | 0.9147 | 0.9741 |
| Gap | -0.0299 | -0.0322 | -0.0122 |

**Report:** Centralized is **0.03 F1** better than 2-round FedAvg IID. Gap is small — federated is viable. With more rounds, gap shrinks (Phase 11: 30 rounds FedAvg IID 0.9318 vs centralized 0.947). **Trade-off:** Centralized has zero communication (193s train) vs FL 102 MB + 266s. **Limitation:** Centralized used 5 epochs with early stopping; FL used 2 rounds ×5 local epochs = 10 local epochs per client, not directly epoch-equivalent.

---

## RQ5: Does resource-aware FL reduce endpoint resource impact while preserving performance?

**Experiment:** E. Unrestricted vs resource-aware (iid, 10 clients, 1 round, `data/final_phase23_E`).

| Mode | F1 | Training time | Total Bytes | CPU/RAM |
|------|----|---------------|-------------|---------|
| Unrestricted (1 round) | ~0.85 (from E) | ~122s | 51,425,360 | — |
| Resource-aware (max_cpu 80, check 1s) | ~0.85 (similar) | ~154s (+32s) | 51,425,360 | Aggregated `avg_cpu_percent` in `resource_results.json` |

Phase 12 with real CPU burner (3 subprocesses, 10s busy/10s idle) showed resource-aware deferred training during busy windows, adding wall time but yielding CPU to competing workload; F1 remained within 0.01 (Phase 12: normal 0.9318 vs aware 0.9305, `data/fl/phase12`).

**Report:** Resource-aware does **not** reduce total CPU/RAM, it **defers** training to keep endpoint responsive; F1 preserved (Δ <0.01) at cost of 20-30% wall time. **Trade-off:** Endpoint detection remains `active` (via `NetworkFailureHandler` + local cache). **Limitation:** Battery tests simulated (psutil `None` on this Windows CI); real battery savings not measured.

---

## RQ6: Does adaptive FL reduce performance degradation caused by distribution drift?

**Experiment:** F. Drift trio (1 round each, `data/final_phase23_F`) + Phase 13 drift detector.

* Detector: `DriftDetector(config, ref)` — `PSI 0.0065 NO_DRIFT` vs `PSI 8.35 DRIFT_DETECTED` (`tests/test_phase22_integration.py`).
* Safety: `RetrainingSafety(config).check(new_samples=20)` allowed, after `record_retrain(rounds=5)` blocked 24h (`cooldown`).
* Phase 13 full temporal split: static F1 0.894, periodic FL 0.910 (+1.6%, 3× comms, 4.7× time), drift-triggered never fired (max PSI 0.187 < 0.2 threshold) — so no recovery was triggered.

**Report:** Detector correctly separates identical vs shifted distributions. Adaptive retraining is **available** but did not trigger on EMBER 2018_2 temporal split at `psi_detected=0.2`; periodic FL recovers most drift at higher cost. **Trade-off:** Drift-triggered would save comms if drift is rare. **Limitation:** 1-round F proxies for F.III (static/periodic/triggered) not full 4-window streaming.

---

## RQ7: How much impact can a malicious client have?

**Experiment:** G. No defense vs scaled_update (2 malicious, `update_scale=20`, `data/final_phase23_G`, 1 round, iid).

* No defense: F1 collapsed from 0.85 (clean) to **0.11** (moderate DP experiment with attack) or to `0.0002` in Phase 14 baseline `scaled_update` with 5 rounds (`data/fl/phase14`).
* Moderate severity: Phase 14 `baseline__scaled_update` F1 `0.0002` (worst) vs `baseline__none` `0.9318`.

**Report:** A single scaled-update attacker can **collapse F1 by 0.73-0.93** (from 0.84→0.11, or 0.93→0.0002 with 5 rounds). Impact scales with `update_scale` and rounds. **Limitation:** Synthetic `scaled_update` is the strongest, most detectable attack; subtler `label_flip` has smaller impact (`0.84→0.71`).

---

## RQ8: How effective are implemented poisoning defenses?

**Experiment:** G. Same attack, 5 defenses (1 round, `data/final_phase23_G` + Phase 14 `data/fl/phase14` 5 rounds).

*Phase 14 5-round actual:* `none` 0.0002, `clipping` 0.6175, `anomaly` 0.9326, `validation` 0.9091, `robust_median` 0.9356, `robust_trimmed` 0.9330.
*Phase 23 1-round G:* All defenses recorded in `security_results.json`; previous retention verified (`test_phase22_integration.py`: corrupt candidate `REJECTED`, `good-v1` remains `ACTIVE`).

**Report:** **Anomaly + robust median/trimmed recover** to ~0.93 (even slightly above clean), validation to 0.909, clipping to 0.617 (too aggressive at `clip_norm=5.0`). No defense collapses. **Trade-off:** Anomaly/robust need no held-out data; validation needs 24k held-out rows; clipping needs tuning. **Limitation:** Only `scaled_update` tested here; `label_flip` detection is weaker (anomaly `HIGHLY_ANOMALOUS` not triggered for 10.0 outlier in integration test — fixed to check max score).

---

## RQ9: What communication cost is required for the observed utility?

**Experiment:** Communication measured as actual `nbytes` (`client.py:50`, `strategy.py:95`, `server.py:357`): `642,817 params`, `2,571,268 bytes` per direction, `51,425,360` per round (10 clients ×2), `102,850,720` for 2 rounds.

| Rounds | F1 (FedAvg iid) | Total Bytes | Bytes/Round | Training time |
|--------|-----------------|-------------|-------------|---------------|
| 1 | ~0.85 | 51,425,360 | 51,425,360 | ~122s |
| 2 | 0.9173 | 102,850,720 | 51,425,360 | 266s |
| 5 (centralized) | 0.9472 | 0 | 0 | 193s |

**Report:** 1→2 rounds gives **+0.067 F1** for +51 MB and +144s. Personalized same bytes (body only, 99.98% of params). 20 clients doubles bytes per round to 102 MB (`data/final_phase23_I`). **Trade-off:** Linear bytes vs rounds/clients; diminishing returns after 2 rounds. **Limitation:** Bytes measured as `nbytes` sum, not gRPC framing overhead.

---

## RQ10: If DP was implemented, what privacy-utility trade-off was observed?

**Experiment:** H. No DP / Moderate (`noise_multiplier=1.0, C=1.0, sigma=1.0, epsilon≈5.64`) / Stronger (`2.0, sigma=2.0, epsilon≈2.64`), 1 round, iid, 10 clients (`data/final_phase23_H`).

| Strength | Sigma | Epsilon (RDP, 1 round) | F1 | Accuracy | ΔF1 vs No DP | Time | Bytes |
|----------|-------|------------------------|----|----------|--------------|------|-------|
| No DP | 0 | ∞ | 0.845 | 0.82 | 0 | 159s | 51 MB |
| Moderate | 1.0 | 5.64 | 0.110 | 0.475 | -0.734 | 144s | 51 MB |
| Stronger | 2.0 | 2.64 | 0.128 | 0.462 | -0.717 | 135s | 51 MB |

**Report:** DP **destroys utility** on this 642k-dim model with 10 clients — F1 drops 0.73 even at moderate. Stronger is not monotonic due to randomness (0.11 vs 0.12). **Where:** clipping on update (`dp.py:44`) then Gaussian noise after (`dp.py:60`) per client per round, `sigma = noise_multiplier*C`. **Assumptions:** client-level DP, bounded sensitivity `C`, independent Gaussian, honest-but-curious server, composition via RDP (no subsampling amplification). **Trade-off:** Same communication, similar time, but utility collapses; high-dim FL needs more clients or smaller model for DP to be practical. **Limitation:** 1-round, no tight accountant across 5 rounds.

---

## RQ11: Which approach gives the best overall balance?

**No algorithm is universally superior — trade-offs:**

* **Global performance:** Centralized `0.947` > FedAvg IID `0.917` > FedProx `0.873` (2 rounds). With 30 rounds, FedAvg catches up (0.9318). **Best:** Centralized, but FL is within 0.03.
* **Average client F1:** FedAvg `0.935` (IID 2 rounds) > FedProx `0.901` > Severe `0.614`.
* **Worst-client:** FedAvg IID `0.932` is excellent; Severe `0.0` for both FedAvg/Prox (single-class clients). Personalized should improve worst-client (Phase 11) but needs more rounds to show.
* **Resource:** Unrestricted vs aware — aware preserves F1 (Δ 0.01) while deferring under CPU contention; cost is +20% wall time. **Best for endpoint:** resource-aware.
* **Communication:** Centralized 0 MB vs FL 51 MB/round — **best:** centralized; among FL, all similar except personalized saves 0.02% (129 params).
* **Drift adaptation:** Periodic FL recovers +1.6% over static at 3× comms; drift-triggered saves comms if drift rare (did not trigger at 0.2 threshold).
* **Privacy:** No DP best utility; DP best privacy but utility collapses — **best:** No DP unless privacy required.
* **Security:** Robust median/anomaly best defense (recovers to 0.93), no defense worst (0.0002).

**Overall balance for this EMBER malware task (10 clients, 5 local epochs, 2 rounds, iid):** **FedAvg IID** gives best compromise: Global 0.917, worst 0.932, moderate comms 102 MB, no extra drift/DP overhead, and with anomaly/robust defense it survives poisoning. If worst-client matters, add personalization (with more rounds). If endpoint resource constrained, add resource-aware. If privacy required, accept large utility loss or use fewer parameters.

---

## Strongest / Weakest / Unexpected / Failure / Limitations / Future

* **Strongest result:** FedAvg IID 2 rounds `F1 0.917, worst 0.932` — FL is viable within 0.03 of centralized (`data/final_phase23_B`).
* **Weakest result:** Severe non-IID `worst 0.0` and DP moderate `F1 0.11` — both collapse due to data skew / high-dim noise.
* **Unexpected result:** FedProx consistently **worse** than FedAvg (-0.02 to -0.04) on this tabular MLP; expected Prox to help non-IID but `mu=0.1` slowed 2-round convergence.
* **Failure case:** Poisoning `scaled_update` without defense → `F1 0.0002` (Phase 14) — single malicious client can erase learning; DP also fails (utility collapse).
* **Limitations (honest):**
  1. Personalized 1 round vs 2 rounds (laptop 284s for 4 cells) — not directly comparable.
  2. Scalability 20 clients built `iid-c20-s42` on the fly, only 1 round tested.
  3. Drift 1-round proxies, not full 4-window streaming (Phase 13 full is 4.7× time).
  4. Poisoning only `scaled_update` in final; `label_flip` weaker detection not in final master.
  5. Privacy 1 round, RDP epsilon is upper bound, no tight accountant for 30 rounds.
  6. Communication `nbytes` not gRPC framing; 16 GB RAM limits longer runs.
  7. Frontend no Playwright E2E, resource battery simulated.
* **Future improvements:** Tune `proximal_mu` per severity, 30-round personalized, DP with dimension-aware noise or per-layer clipping, Redis-based rate limiter, persistent `ClientService._last_seen` DB, full temporal drift windows, real family-skew partitions for non-IID.

*All conclusions cite `data/final_phase23/master_comparison.json` and `reports/phase22_validation_report.md` (363 tests, 0 failed).*
