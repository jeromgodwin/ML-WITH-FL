# Final Product Quality Review — Enhancement 28

**Treat FedShield as a serious technical demonstration. 371 tests passed, 31 warnings.**

---

## MUST FIX (before demo)

* **M1 — Rate limiter in-memory** `backend/security/rate_limiter.py:11` — per-process, not distributed. Blocks bypass with multiple replicas. *Fix:* Redis for Tier 2.
* **M2 — Client last-seen in-memory** `backend/services/client_service.py:11` — lost on restart. *Fix:* DB persistence.
* **M3 — Self-signed TLS without CA** `src/federated/network/tls.py:24` fallback `_create_unverified_context`. *Fix:* Require `ca_cert` for `host != 127.0.0.1` when `secure=true`.
* **M4 — 16 GB OOM on 30-round full matrix** `src/federated/experiments/engine.py:230` — needs per-cell fresh process (Phase 14 fix) for final 29-run matrix.
* **M5 — DP high-dim collapse** `src/federated/privacy/dp.py:44` — 642k dim F1 0.845→0.11. *Fix:* Document as known, offer per-layer clipping.

---

## SHOULD FIX (strongly recommended)

* **S1 — Frontend no Playwright E2E** `frontend/src/components/*` — only `vite build` verified. Add Playwright for Tier 1.
* **S2 — Battery `None` on CI** `src/endpoint/resource/monitor.py` — low-battery never tested on hardware. Manual laptop test.
* **S3 — Drift 1-round proxies in final** `data/final_phase23` — full 4-window is 4.7× time. Add 30-round drift e2e.
* **S4 — Personalized 1 vs 2 rounds** `data/final_phase23_D1` — not directly comparable to FedAvg. Add 5-round personalized.
* **S5 — Scalability 20 clients built on-the-fly** `data/iid-c20-s42` — not pre-validated at 30 rounds.
* **S6 — README gaps** `backend/README.md` `frontend/README.md` — 4 lines each. Expand with `uvicorn` + `npm run dev` health report.

---

## NICE TO HAVE

* **N1 — `torch` duplication** `pyproject.toml:11` vs `requirements.txt:1`
* **N2 — `logs/` not rotated** `fedshield/config.py:268`
* **N3 — `clients.json` polling per call** `src/federated/network/auth.py:28` — cache + mtime
* **N4 — `VITE_API_BASE` not configurable** `frontend/src/api.js:1` — add env var for LAN/Internet
* **N5 — Empty `__init__.py`** `src/federated/privacy/__init__.py`

---

## Category Scores

* **Usability:** Good (11-tab dashboard, no file-upload, health report) — needs Playwright
* **Security:** Good (TLS, auth, validation, replay, no raw files) — needs CA enforcement, Redis limiter
* **Reliability:** Good (7 states, retry/backoff, rollback, offline queue) — needs DB persistence
* **Performance:** Good (persistent model, chunked gather, 51 MB/round) — needs `gather()` reuse for centralized (done in Phase 15)
* **Research:** Strong (29 fair runs, no cherry-picking, 11 RQs) — needs 30-round personalized + per-layer DP
* **Deployment:** Good (dev/LAN/Internet via `configs/deployment_profiles.yaml`, one-laptop sim) — needs CA certs

**Overall:** Demo-ready for one-laptop → LAN, with documented limitations. Fix **MUST 5** before Internet public demo.
