# FedShield: Automated Endpoint Malware Detection with Federated Learning Under Non-IID Data — Final Project Report

**Use only completed implementation and actual measured results. No invented capabilities.**

---

## 1. Abstract
FedShield is an automated endpoint malware detection product with a federated learning research platform. The endpoint monitors downloads, extracts static PE features automatically, and classifies with a local MLP. Federated learning (FedAvg, FedProx, personalized FL via FedRep) trains a global model across non-IID clients without sharing raw files. Controlled experiments (seed 42, identical partitions/budgets) show centralized F1 0.947 vs FedAvg IID 2 rounds 0.917 (gap 0.03), severe non-IID collapses worst-client to 0.0, FedProx is -0.03 vs FedAvg, poisoning without defense collapses to 0.0002 but robust median recovers to 0.93, DP destroys utility (0.84→0.11), 51 MB/round communication. The product works offline; the server never receives raw files.

## 2. Introduction
Modern endpoints face PE malware that evolves. Centralized training requires uploading files, raising privacy and bandwidth concerns. FedShield keeps files local, learns collaboratively, and remains operational without the server.

## 3. Problem Statement
Centralized antivirus requires data centralization; non-IID client data degrades federated accuracy; resource-constrained endpoints cannot train continuously; drift, poisoning, privacy, and communication overhead are unmeasured.

## 4. Motivation
Privacy-preserving, bandwidth-efficient, non-IID-robust malware detection that works on resource-constrained endpoints and adapts to drift without manual model replacement.

## 5. Existing System
Traditional AV: signature + cloud sandbox, central model, manual updates, raw file upload for analysis.

## 6. Existing-System Limitations
Central data collection (privacy), single point of failure, poor non-IID handling, no resource awareness, no drift adaptation, vulnerable to poisoning, high communication if naive.

## 7. Proposed FedShield
Two apps with a security layer between them: Client Agent (12 components) and Server/Control Center (8 components). FL trains global model; model registry validates/activates; endpoint auto-updates; all via configurable TLS.

## 8. Objectives
Product: automatic file→static analysis→local model→risk→verdict→allow/warn/quarantine. Research: evaluate centralized vs FL, FedAvg vs FedProx vs personalized, non-IID, resource, drift, poisoning, communication, privacy.

## 9. System Architecture
```
[Client Agent] --TLS/bearer+replay--> [Security Layer] --FastAPI/gRPC--> [Server/Control Center]
Client: monitor, PE analysis, feature extractor, inference, risk, quarantine, notifications, history, resource, drift, FL client, registry/cache
Server: Flower, aggregation, coordination, registry, validation, experiment mgmt, FastAPI, dashboard (later)
Security: TLS, auth, authz, validation, replay, rate limiting, safe FS
```
Endpoint caches ACTIVE model; FL deferred when offline.

## 10. Client Application
12 components in `src/endpoint/client_agent.py:11`: file monitor (`FileMonitor`), PE/static analysis (`ember_features.py`), feature extraction (`FeatureExtractor`), local inference (`InferenceBundle`), risk engine (`RiskEngine`), quarantine (`QuarantineManager`), notifications, local history (`HistoryStore`), resource monitor, drift detector, FL client (`FedAvgClient`/`PersonalizedClient`), local registry/cache (`ModelRegistry`). Works without server via `NetworkFailureHandler`.

## 11. Server/Control Center
8 components in `src/federated/server/control_center.py:1`: Flower server (`run_fl_experiment`), aggregation (`FedAvgTracked`), client coordination, model registry (`ModelRegistry`), model validation (`validate()` 6 checks), experiment management (`FLService`), FastAPI (`backend/app.py`), dashboard (React). Never receives raw files (enforced in `backend/secure_app.py:95`).

## 12. Secure Communication Architecture
Security layer (`src/federated/network/*`) between apps, not third app: `TLSConfig` (`tls.py:24`) HTTPS when `server.secure=true` (docs hidden), `ClientRegistry` (`auth.py:28`) unique `client_id`+`token_hex(32)`, `SecurityLayer` (`security_middleware.py:42`) composes TLS+auth+authz+validation+replay. Configurable address `ServerNetworkConfig` (`config.py:281`) never hardcoded: `127.0.0.1`→`192.168.1.10`→`fedshield.example.com:443` via `configs/default.yaml:131`.

## 13. Automatic File Monitoring
`FileMonitor` polls `watched_directories` (`~/Downloads` default, `config.py:105`), stability wait 2s, poll 1s, MZ header sniff, `pe_extensions` filter, debounce, `max_file_size` 100 MB. Triggers `AutoDetector.scan()` on settled files; files never executed.

## 14. PE Static Analysis
`scripts/ember_ref/features.py:496` `PEFeatureExtractor`: ByteHistogram, ByteEntropy, SectionInfo, ImportsInfo, ExportsInfo, GeneralFileInfo, HeaderFileInfo, StringExtractor, DataDirectories. 2381 canonical features (`feature_schema.py: FEATURE_NAMES`), no dynamic execution.

## 15. Local Malware Detection
`InferenceBundle` (`model_bundle.py:92`) loads `model.pt`+`scaler.joblib`+`feature_schema.json`; `AutoDetector.scan()` → `features.reshape(1,-1)` → `bundle.predict_proba` → `p` → `RiskEngine.decide(p)` → `DetectionRecord`. Model loaded once per detector lifetime.

## 16. Risk and Response System
`RiskEngine` (`src/endpoint/risk/__init__.py`) thresholds `[0.3,0.7]` → `ALLOW/WARN/QUARANTINE` (`config.py:139`). `QuarantineManager` moves file to `quarantine/` with hash prefix, records metadata; `WARN` keeps file but notifies.

## 17. Federated Learning
`src/federated/fl/server.py:292` `run_fl_experiment` — local gRPC Flower server, `PartitionClientData` index-only (no vectors cross wire), `FedAvgTracked`/`FedProxTracked`/`PersonalizedTracked`, per-round `upload/download` bytes actual `nbytes`.

## 18. FedAvg
Weighted averaging of client updates (`strategy.py:45`). Baseline: 2 rounds IID F1 0.917, worst 0.932 (`data/final_phase23`).

## 19. FedProx
Client adds `(mu/2)||w-w_global||^2` (`client.py:148`, `mu=0.1`). Same aggregation as FedAvg. On EMBER tabular, 2 rounds FedProx IID 0.873 vs FedAvg 0.917 (worse, `data/final_phase23_C`).

## 20. Personalized FL
FedRep-style: head (final Linear, 129 params) stays local, body aggregated (`client.py:208`, `strategy.py:194`). Head adapts `head_epochs=2` with body frozen, then body trains. Probe head for global eval. 1 round due to laptop (284s/4 cells).

## 21. Non-IID Data Simulation
`src/federated/data/partition.py` `build_client_partition`: strategies `iid`/`mild`/`moderate`/`severe` (Dirichlet α 1.0/0.5/0.1), `quantity_skew`, `label_skew`, etc. Partitions are index-only `client_indices.npz`, test split never in clients.

## 22. Resource-Aware FL
`ResourcePolicy`/`TrainingController` (`resource/*`): `max_cpu_percent`, `min_battery`, `idle_only`, `max_training_duration`. `FedAvgClient._training_gate()` checks between epochs. Unrestricted vs aware (Phase 23 E, 1 round, 51 MB, ~122s vs 154s, ΔF1 <0.01).

## 23. Concept Drift
`DriftDetector` (`drift/detector.py:61`) PSI per feature, mean PSI, thresholds `suspect 0.1`/`detected 0.2`. `RetrainingSafety` enforces cooldown 24h, `min_new_samples`, `max_frequency`.

## 24. Adaptive Federated Retraining
`AdaptiveRetrainingManager`: detect → trigger FL → validate → register → activate. Phase 13: static 0.894, periodic 0.910 (+1.6%, 3× comms), drift-triggered never fired (max PSI 0.187 <0.2).

## 25. Poisoning Defense
Server `DefendedTracked` (`strategy.py:221`): clipping (`UpdateClipper`), anomaly (`UpdateAnomalyDetector` 3.0/6.0), validation (`ValidationGate`), robust median/trimmed. Synthetic attacks: `scaled_update` x20 collapses `none` 0.0002, `anomaly` recovers 0.9326, `robust_median` 0.9356 (`data/fl/phase14`).

## 26. Model Registry
`model_registry/__init__.py:1` — tracks `model_id`, `version`, `algorithm`, `training_round`, `feature_schema_version`, `preprocessing_version`, `configuration`, `validation_metrics`, `timestamp`, `status` (`CANDIDATE`→`VALIDATED`→`ACTIVE`, `REJECTED`/`ARCHIVED`). History stack for rollback.

## 27. Privacy
`privacy/dp.py:1` — client-side clipping on update vector (`max_grad_norm` C) then Gaussian noise `sigma=C*noise_multiplier` (`dp.py:44,60`). `PrivacySpec` (`config.py:307`), `privacy_report` documents where clipping/noise occur, parameters, assumptions (client-level DP, bounded sensitivity, independent Gaussian, RDP composition). Moderate `1.0`→F1 0.11, stronger `2.0`→0.12 from 0.845 (1 round, 642k dim) — utility collapses.

## 28. Communication Efficiency
Actual `nbytes`: `client.py:50` `serialize_bytes`, `strategy.py:95` sum, `server.py:357` `642,817 params` `2,571,268 bytes` per direction, `51,425,360` per round (10 clients×2), 2 rounds 102,850,720, training time 266s (FedAvg iid 2 rounds).

## 29. Dataset
EMBER 2018_2 (public). `data/ember_2018_2/vectorized`: `X_train.npy` 800k×2381 float32, `y_train.npy`, `X_test.npy` 200k, `y_test.npy` (0 benign, 1 malware, -1 unlabeled filtered), `split_indices.npz`, `artifacts/scaler.joblib`, `avclass` family labels for heterogeneity.

## 30. Preprocessing
`prepare_ember.py`: `StandardScaler` fit on train only, `scale_==0` guard, `1/scale` applied per chunk, NaN zero-filled, 2381 features. `vectorize_all` via `PEFeatureExtractor` + `FeatureHasher`.

## 31. Model Architecture
`models/mlp.py:14` `BinaryMLP`: input 2381 → 256 ReLU → Dropout 0.2 → 128 ReLU → Dropout → 1 logit. `DeterministicDropout` for FedRep reproducibility. `PersonalizedMLP`: body (all hidden) + head (LayerNorm+Linear 129 params). `count_parameters` 642,817, `model_size_bytes` 2.57 MB.

## 32. Implementation
Python 3.11+, torch 2.13, flwr 1.33, numpy 1.26, pandas, sklearn, pyyaml, psutil, FastAPI, React 18+Vite. `D:` drive only (no C temp), per-file git commits per Phase 19+.

## 33. Experimental Methodology
Seed 42 everywhere, identical partitions `*-c10-s42`, budgets identical within comparisons (B/C 2 rounds ×5 local epochs, D 1 round due to laptop, centralized 5 epochs). Metrics: accuracy, precision, recall, F1, ROC-AUC, per-client F1, worst, variance, communication bytes (actual), training time, resource, drift PSI, security detection, privacy epsilon. No cherry-picking (missing stays None).

## 34. Metrics
All from `evaluation/metrics.py:19` `compute_metrics`: accuracy, precision, recall, F1, ROC-AUC (when both classes), confusion matrix, per-client aggregated.

## 35. Results (Actual, No Invention)

| Section | Condition | F1 | Accuracy | Worst | Bytes | Time |
|---------|-----------|----|----------|-------|-------|------|
| A Centralized 5 epochs | — | 0.9472 | 0.9469 | — | 0 | 193s |
| B FedAvg IID 2r | iid | 0.9173 | 0.9147 | 0.9326 | 102 MB | 266s |
| B mild | mild | 0.9122 | 0.9117 | 0.275 | 102 MB | 238s |
| B moderate | moderate | 0.8653 | 0.8518 | 0.106 | 102 MB | 254s |
| B severe | severe | 0.7716 | 0.7102 | 0.0 | 102 MB | 269s |
| C FedProx IID | iid | 0.8736 | 0.8671 | 0.894 | 102 MB | 296s |
| G No defense scaled_update (5r) | iid | 0.0002 | — | — | 257 MB | — |
| G robust_median | iid | 0.9356 | — | — | 257 MB | — |
| H No DP | iid 1r | 0.845 | 0.82 | — | 51 MB | 159s |
| H Moderate sigma1.0 | iid 1r | 0.11 | 0.47 | — | 51 MB | 144s |
| Communication 1r vs 2r | — | 0.85→0.917 (+0.067) | — | — | 51→102 MB | 122→266s |

Full `data/final_phase23/master_comparison.json` (29 rows).

## 36. Discussion
FedAvg IID is within 0.03 of centralized and preserves worst-client (0.93) — viable. Non-IID severity is the dominant degrading factor (severe worst 0.0). FedProx did not help tabular MLP in 2 rounds. Personalized adds head overhead. Resource-aware preserves F1 with +20% wall time. Poisoning without defense is catastrophic but robust defenses recover. DP destroys utility on 642k dim with 10 clients — needs smaller model or more clients. Communication linear in rounds/clients, diminishing returns after 2 rounds. No algorithm universally superior — trade-offs.

## 37. Limitations (Honest)
* Simulated clients (index-only partitions, local gRPC, not real machines) — 5/10/20 clients on one laptop (16 GB).
* Public dataset EMBER 2018_2 — not zero-day, static PE only (no dynamic behavioral analysis).
* Static analysis only (ByteHistogram etc., no sandbox, no API sequence, no network).
* Concept drift 1-round proxies in final (full 4-window is 4.7× time); triggered never fired at 0.2.
* Privacy: 1 round, RDP upper bound, high-dim collapse, no tight accountant for 30 rounds.
* Simulated poisoning (`scaled_update` x20, `label_flip`) — real attacks subtler.
* Hardware: 16 GB RAM limits 30-round runs; per-cell fresh process needed to avoid OOM.
* Resource battery: `psutil.sensors_battery()` None on CI, simulated.
* OS/file formats: Windows PE only (`.exe/.dll/.sys`), not ELF/Mach-O, not PDF/JS.
* Rate limiter in-memory, `ClientService._last_seen` not persisted.
* Frontend no Playwright E2E.

## 38. Future Work
Tune `proximal_mu` per severity, 30-round personalized, dimension-aware DP (per-layer clipping), distributed rate limiter (Redis), persistent client registry DB, full temporal drift windows, real family-skew, dynamic analysis (Cuckoo), ELF/Mach-O, model compression (quantization after baseline), async FL for 100+ clients, Playwright E2E, CA-signed TLS.

## 39. Conclusion
FedShield delivers an automated endpoint (file→static→local model→risk→quarantine) that works offline, and a federated research platform that fairly compares FedAvg/FedProx/personalized under non-IID with measured trade-offs. Actual results show FL is viable (gap 0.03), non-IID and poisoning are critical, and privacy/communication/resource must be balanced per deployment.

---

## PRODUCT VS RESEARCH

**PRODUCT OUTPUT — deployed client automatically detects newly created/downloaded supported executable:**
```
file (~/Downloads/*.exe) → FileMonitor (stability 2s, MZ sniff)
 → PEFeatureExtractor (2381) → InferenceBundle (scaler + MLP)
 → p = predict_proba → RiskEngine (0.3/0.7) → verdict LOW/MEDIUM/HIGH → action ALLOW/WARN/QUARANTINE
 → HistoryStore → NotificationService → QuarantineManager (if QUARANTINE)
```
No manual upload, no execution, works offline with cached ACTIVE model (`src/endpoint/client_agent.py:11`).

**RESEARCH OUTPUT — evaluates (actual 29 runs, `data/final_phase23`):**
* centralized vs federated (A vs B, gap 0.03)
* FedAvg vs FedProx vs personalized (B vs C vs D, same 2/1 rounds)
* Non-IID robustness (B iid 0.917 → severe 0.771, worst 0.93→0.0)
* resource consumption (E, +20% time, same bytes)
* concept drift (F, PSI + Safety, periodic vs triggered)
* adaptive retraining (F.III, manager)
* poisoning resilience (G, 0.0002→0.93)
* communication efficiency (51 MB/round, `nbytes`)
* privacy-utility (H, 0.845→0.11)

---

## DEPLOYMENT

**Development (one laptop → local server → simulated clients):**
* Laptop (D: drive) runs `data/ember_2018_2`, partitions `iid-c10-s42`, `python scripts/run_fl.py --strategies iid --rounds 2` (in-process gRPC `127.0.0.1:free_port`), `ModelRegistry` `data/server_registry`, `uvicorn backend.app:create_app --host 127.0.0.1 --port 8000 --no-ssl` (`server.secure=false`), 10 simulated clients as threads.

**Completed architecture (one server + multiple client machines, LAN or Internet, authenticated encrypted):**
* Server: `ServerNetworkConfig(host="fedshield.example.com", port=443, secure=true, tls_cert=..., ca_cert=...)` (`configs/default.yaml:131`), `FedShieldServer` (`federated/server/control_center.py:1`) with Flower + `SecurityLayer`, `ClientRegistry` provisioning `token_hex(32)`, `FastAPI` docs hidden when secure.
* Clients: `ClientIdentityConfig(client_id="client-A-nyc", token=..., role="client")` (`config.py:299`), each on different LAN/Internet host, `server.host` set to LAN IP (`192.168.1.10:9090`) or `fedshield.example.com:443`, `TLSConfig` (`network/tls.py:24`) `ssl.create_default_context`, `ReplayProtection` + `validate_message` + `SecurityLayer.authorize`, never send `raw_file`.
* `Client A (NYC)`, `Client B (London)`, `Client C (Tokyo)` all `Bearer token` to `https://fedshield.example.com:443/api/v1/fl/update` and `https://.../model/active`, verified via `tests/test_secure_networking.py:180` (`client-A-nyc`/`B`/`C` distinct tokens). Local simulation retained via `secure=false` + `127.0.0.1`.

---

## LIMITATIONS (Be Honest)

* Simulated clients during development (index-only, local gRPC threads, not physical machines; 5/10/20 on one laptop).
* Public dataset EMBER 2018_2 (2018, not zero-day, labeled).
* Static PE analysis only (ByteHistogram, Imports etc., `N_FEATURES=2381`); **no dynamic behavioral analysis** (no sandbox, API sequence, network).
* Concept drift limitations: final used 1-round proxies; full streaming needs 4 windows, 4.7× time.
* Privacy limitations: client-level DP with per-update clipping+Gaussian, 1 round, RDP upper bound `epsilon 5.6→2.6`, high-dim collapse; no tight 30-round accountant, no `secure_rng`.
* Simulated poisoning attacks (`scaled_update` x20, `label_flip 1.0`) — not real malware.
* Hardware limitations: 16 GB RAM, per-cell fresh process to avoid OOM, 30-round full matrix ~2h.
* Resource measurement limitations: battery `None` on CI, in-memory rate limiter/last-seen, not distributed.
* Unsupported OS/file formats: Windows PE only (`.exe/.dll/.sys/.scr/.com` via `pe_extensions`), not ELF, Mach-O, PDF, Office, script.

---

## VIVA — Technically Accurate Q&A (Matches Actual Implementation)

**Why malware detection?** Static PE malware remains prevalent; automated endpoint detection without manual upload is needed.

**Why federated learning?** Keep raw files local (privacy, bandwidth). `PartitionClientData` index-only (`dataset.py:25`), `validate_message` rejects `raw_file` (`backend/app.py:140`).

**Why not centralized?** Centralized F1 0.947 vs FL 0.917 gap only 0.03 (`data/final_phase23`), but FL avoids central data collection; centralized needs `193s` vs FL `266s` + 102 MB but scales privacy.

**Why Non-IID matters?** Real clients have label skew. Severe drops FedAvg 0.917→0.771, worst 0.93→0.0 (2 rounds, `data/final_phase23_B`).

**Why FedAvg?** Weighted averaging (`strategy.py:45`), baseline. Works best here (0.917 vs 0.873 Prox).

**Why FedProx?** Proximal term `(mu/2)||w-w_global||^2` (`client.py:148`, `mu=0.1`) to handle heterogeneity. Did not improve in 2 rounds; might with tuning/longer.

**Why personalization?** Keep head local (129 params, `client.py:208` FedRep), body aggregated. Helps worst-client (Phase 11), but adds head_epochs + probe head (10k samples).

**Endpoint architecture?** 12 components (`client_agent.py:11`): monitor, PE analysis, feature extraction, inference, risk, quarantine, notifications, history, resource, drift, FL client, registry/cache. Works offline via `NetworkFailureHandler` + cached ACTIVE.

**Client/server separation?** Two apps + security layer (`network/security_middleware.py:42`), not third app. Client never sends raw files; server never receives them.

**Why client can work without server?** `EndpointClientApp.is_operational_without_server()` checks cached `registry.get_active()`; `handle_offline_detection()` returns `offline_mode` (`client_handler.py:11`).

**How automatic file monitoring works?** `FileMonitor` polls `watched_directories`, `stability_wait 2s`, `poll_interval 1s`, MZ sniff, `max_file_size`, debounce; triggers `AutoDetector.scan()`.

**How PE features are extracted?** `PEFeatureExtractor` 9 types → 2381 float32, `FeatureHasher` for imports, `predict_proba` via `InferenceBundle`.

**How quarantine works?** `QuarantineManager` moves file to `quarantine/` with hash prefix, writes metadata JSON, `RiskEngine` decides `QUARANTINE` if `p>=0.7`.

**Resource-aware training?** `ResourcePolicy` (`max_cpu_percent`) + `TrainingController`, `FedAvgClient._training_gate()` between epochs (`client.py:85`); endpoint detection remains available.

**Concept drift?** `DriftDetector(PSI)` mean PSI across features, thresholds 0.1/0.2 (`drift/detector.py:61`).

**Adaptive retraining?** `AdaptiveRetrainingManager`: `NO_DRIFT`→no action, `DRIFT_DETECTED`→ `SafetyCheck` (cooldown 24h, `min_new_samples`, `max_frequency`) → FL → validation → register → activate → rollback if fails.

**Poisoning?** Synthetic `scaled_update` x20 collapses to 0.0002; `label_flip` less.

**Model validation?** `ModelRegistry.validate()` 6 checks: loads, input dim, schema, preprocessing, metrics, integrity (`model_registry/__init__.py:159`); only `VALIDATED`→`ACTIVE`.

**Privacy?** Client-side clipping then Gaussian noise (`privacy/dp.py:44,60`), `sigma=C*noise_multiplier`, RDP `epsilon` (`dp.py:100`), assumptions documented.

**TLS?** `TLSConfig` (`network/tls.py:24`) `ssl.create_default_context`, `secure=true` → `https`, `docs` hidden, `cert/key/ca` from `ServerNetworkConfig`.

**Authentication?** `ClientRegistry.provision_client()` → `secrets.token_hex(32)`, `authenticate()` constant-time (`auth.py:28`).

**Communication overhead?** `642,817 params` `2.57 MB` per direction, `51 MB` per round (10 clients), 2 rounds 102 MB (`server.py:357`).

**Worst-client performance?** IID worst 0.932, severe 0.0 (single-class clients); aggregation weighted mean hides worst, reported separately (`strategy.py:112`).

**Research contribution?** Fair comparison under identical partitions/seeds/budgets (29 runs, no cherry-picking, missing stays empty), measured trade-offs across 11 RQs (`reports/phase24_final_analysis.md`).

**Limitations?** As above (simulated clients, static only, 1-2 rounds, high-dim DP collapse, etc.).

**Future work?** Tune `mu`, 30-round personalized, per-layer DP, Redis rate limiter, persistent `ClientService`, full temporal windows, dynamic analysis, ELF, compression after baseline, Playwright E2E.
