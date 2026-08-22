# Secure Aggregation Investigation (Enhancement 13)

**Flower version:** 1.33+ — supports `secaggplus` via `flwr.server.strategy` but requires all clients to implement `get_secagg_config`.

**Threat model:** Honest-but-curious server should not see individual updates, only aggregate. Protects update confidentiality, not poisoning or authentication.

**What is protected:** Individual `ndarrays` before aggregation.

**What is not:** Model poisoning, client identity, availability, final aggregate.

**Dropout assumptions:** Requires >50% clients survive round; dropout handling via Shamir sharing adds 2× communication overhead.

**Overhead:** ~2-3× bytes, +30% CPU for share generation.

**Conclusion:** Correct secure aggregation is possible via Flower `SecAggPlus` but needs client-side `secagg` implementation and synchronous round (all clients). For 10 clients on one laptop, overhead is high and threat model (malicious client) is already handled via poisoning defense. **Documented as future work** for production LAN/Internet deployment — do not fake.

If implemented: `strategy = SecAggPlusFedAvg(...)` with `min_available_clients=10`.
