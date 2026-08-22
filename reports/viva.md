# FedShield Viva — Technically Accurate Q&A (Matches Actual Implementation)

All answers cite `file:line` and measured results. No invented capabilities.

**Why malware detection?** PE malware still prevalent; endpoint needs automatic detection without manual upload. FedShield monitors `~/Downloads` via `FileMonitor` and classifies locally.

**Why federated learning?** Keep raw files local. `PartitionClientData` is index-only (`src/federated/fl/dataset.py:25`), `validate_message` rejects `raw_file` (`backend/app.py:140`), server only sees `Parameters` (`src/federated/fl/server.py:355`).

**Why not centralized?** Centralized F1 0.947 vs FedAvg IID 0.917 (gap 0.03, `data/final_phase23`), but FL avoids central data; gap shrinks with 30 rounds (0.9318).

**Why Non-IID matters?** Real clients skewed. Severe (Dirichlet 0.1) drops FedAvg 0.917→0.771, worst 0.93→0.0 (`data/final_phase23_B`).

**Why FedAvg?** Weighted averaging (`src/federated/fl/strategy.py:45`), baseline best here (0.917 vs FedProx 0.873).

**Why FedProx?** Adds `(mu/2)||w-w_global||^2` (`src/federated/fl/client.py:148`, `mu=0.1`) for heterogeneity. Did not improve in 2 rounds; needs tuning.

**Why personalization?** FedRep head (129 params) stays local (`src/federated/fl/client.py:208`), body aggregated. Helps worst-client, adds head_epochs + probe head (10k).

**Endpoint architecture?** 12 components (`src/endpoint/client_agent.py:11`): monitor, PE analysis, feature extraction, inference, risk, quarantine, notifications, history, resource, drift, FL client, registry/cache.

**Client/server separation?** Two apps + security layer (`src/federated/network/security_middleware.py:42`), not third app. Client never sends raw files.

**Why client works without server?** `EndpointClientApp.is_operational_without_server()` checks cached `ACTIVE` (`src/federated/network/client_handler.py:11`); `handle_offline_detection()` returns `offline_mode`, FL deferred, queue persisted.

**How file monitoring works?** `FileMonitor` polls `watched_directories`, `stability_wait 2s`, `poll_interval 1s`, MZ sniff, `max_file_size` 100 MB.

**How PE features extracted?** `PEFeatureExtractor` 9 types → 2381 features, `FeatureHasher` for imports.

**How quarantine works?** `QuarantineManager` moves to `quarantine/` with hash prefix, JSON metadata; `RiskEngine` thresholds `[0.3,0.7]` → `ALLOW/WARN/QUARANTINE`.

**Resource-aware training?** `ResourcePolicy` + `TrainingController`, `FedAvgClient._training_gate()` between epochs (`src/federated/fl/client.py:85`).

**Concept drift?** `DriftDetector` mean PSI, thresholds 0.1/0.2 (`src/drift/detector.py:61`).

**Adaptive retraining?** `AdaptiveRetrainingManager`: `NO_DRIFT`→no action, `DRIFT_DETECTED`→`RetrainingSafety` (cooldown 24h) → FL → validation → activate/rollback.

**Poisoning?** Synthetic `scaled_update` x20 collapses `none` 0.0002, robust recovers 0.93 (`data/fl/phase14`).

**Model validation?** 6 checks: loads, input dim, schema, preprocessing, metrics, integrity (`src/federated/model_registry/__init__.py:159`); only `VALIDATED`→`ACTIVE`.

**Privacy?** Client-side clipping then Gaussian (`src/federated/privacy/dp.py:44,60`), `sigma=C*noise_multiplier`, RDP `epsilon`, moderate 0.11 vs 0.845.

**TLS?** `TLSConfig` `ssl.create_default_context`, `secure=true`→`https`, docs hidden (`backend/secure_app.py:43`).

**Authentication?** `ClientRegistry.provision_client()` → `secrets.token_hex(32)`, constant-time `authenticate` (`src/federated/network/auth.py:28`).

**Communication overhead?** `642,817` params `2.57 MB` per direction, `51 MB`/round (10 clients) (`src/federated/fl/server.py:357`).

**Worst-client performance?** IID `0.932`, severe `0.0`; reported separately (`src/federated/fl/strategy.py:112`).

**Research contribution?** 29 fair runs (identical partitions/seeds/budgets, no cherry-picking, `data/final_phase23`), 11 RQs (`reports/phase24_final_analysis.md`).

**Limitations?** Simulated clients (16 GB laptop), EMBER 2018_2, static only, 1-2 rounds, DP collapse, simulated poisoning, in-memory rate limiter.

**Future work?** Tune `mu`, 30-round personalized, per-layer DP, Redis, persistent registry, full temporal windows, dynamic analysis, ELF, compression, Playwright.
