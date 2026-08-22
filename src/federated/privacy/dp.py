"""Differential privacy for federated updates (Phase 17).

Implements a technically correct, minimal client-level DP primitive:

- Clipping: per-update L2 clipping on the client after local training.
- Noise: Gaussian mechanism on the clipped update, same shape, independent per client per round.

This is client-level DP (one client's entire local dataset is the neighboring unit),
not per-sample DP. Sensitivity is bounded by max_grad_norm (C). The Gaussian
mechanism with sigma = noise_multiplier * C gives (epsilon, delta)-DP per round
for the released update under the bounded-sensitivity assumption.

Composition across rounds is reported via a simple RDP accountant (best-effort).
If a tight end-to-end epsilon cannot be established, the system reports sigma
as the strength indicator and marks epsilon as an upper-bound estimate.

Assumptions (documented):
- Each client's update vector has L2 sensitivity bounded by C after clipping.
- Noise is sampled i.i.d. Gaussian per coordinate, independent across clients/rounds.
- No subsampling amplification is claimed unless client_fraction < 1.0 (reported separately).
- The server is honest-but-curious; DP protects against inference about a client's
  full contribution in a single round. Cumulative privacy over T rounds degrades
  linearly without additional amplification.
- No cryptographic secure RNG by default; numpy's Generator is used (secure_rng=True would use secrets).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class PrivacySpec:
    """Resolved privacy configuration for one run."""

    enabled: bool = False
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    accounting_mode: str = "rdp"
    seed: int = 42

    @classmethod
    def from_config(cls, cfg: Any) -> "PrivacySpec":
        """Build from fedshield.config.PrivacyConfig or a dict."""
        if cfg is None:
            return cls(enabled=False)
        if isinstance(cfg, dict):
            return cls(
                enabled=bool(cfg.get("enabled", False)),
                noise_multiplier=float(cfg.get("noise_multiplier", 1.0)),
                max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
                delta=float(cfg.get("delta", 1e-5)),
                accounting_mode=str(cfg.get("accounting_mode", "rdp")),
                seed=int(cfg.get("seed", 42)) if "seed" in cfg else 42,
            )
        # Assume dataclass with same fields
        return cls(
            enabled=bool(getattr(cfg, "enabled", False)),
            noise_multiplier=float(getattr(cfg, "noise_multiplier", 1.0)),
            max_grad_norm=float(getattr(cfg, "max_grad_norm", 1.0)),
            delta=float(getattr(cfg, "delta", 1e-5)),
            accounting_mode=str(getattr(cfg, "accounting_mode", "rdp")),
            seed=int(getattr(cfg, "seed", 42)) if hasattr(cfg, "seed") else 42,
        )

    def validate(self) -> None:
        if self.enabled:
            if self.noise_multiplier <= 0:
                raise ValueError(f"noise_multiplier must be >0, got {self.noise_multiplier}")
            if self.max_grad_norm <= 0:
                raise ValueError(f"max_grad_norm must be >0, got {self.max_grad_norm}")
            if not 0 < self.delta < 1:
                raise ValueError(f"delta must be in (0,1), got {self.delta}")

    @property
    def sigma(self) -> float:
        """Noise scale (standard deviation) = noise_multiplier * max_grad_norm."""
        return float(self.noise_multiplier * self.max_grad_norm)

    def strength_label(self) -> str:
        if not self.enabled:
            return "none"
        if self.noise_multiplier >= 1.5:
            return "strong"
        if self.noise_multiplier >= 0.8:
            return "moderate"
        return "weak"


def clip_update(update: np.ndarray, max_norm: float) -> Tuple[np.ndarray, bool, float, float]:
    """L2 clip an update vector to max_norm.

    Returns (clipped, did_clip, original_norm, clipped_norm).
    Clipping occurs on the CLIENT, on the update (params_after - params_before),
    before noise is added (required for bounded sensitivity).
    """
    update = np.asarray(update, dtype=np.float32)
    norm = float(np.linalg.norm(update))
    if norm <= max_norm or max_norm <= 0:
        return update, False, norm, norm
    scale = max_norm / norm
    clipped = (update * scale).astype(np.float32)
    return clipped, True, norm, float(np.linalg.norm(clipped))


def add_gaussian_noise(
    update: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
    secure_rng: bool = False,
) -> np.ndarray:
    """Add Gaussian noise N(0, sigma^2) per coordinate.

    Noise is added on the CLIENT, after clipping, before transmission.
    The noised update has the same shape/bytes as the original.
    When secure_rng=True, entropy is mixed from secrets.token_bytes.
    """
    if sigma <= 0:
        return update
    if secure_rng:
        import secrets
        # Mix secrets entropy into the Generator's state (best-effort)
        try:
            entropy = int.from_bytes(secrets.token_bytes(8), "big")
            rng = np.random.default_rng(entropy ^ int(rng.bit_generator.state["state"]["state"] & 0xFFFFFFFF))
        except Exception:
            pass
    noise = rng.normal(0.0, sigma, size=update.shape).astype(np.float32)
    return (update + noise).astype(np.float32)


def apply_dp_to_update(
    update: np.ndarray,
    spec: PrivacySpec,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Clip + noise one client's update vector (correct order).

    Returns (noised_clipped_update, metrics).
    """
    if not spec.enabled:
        return update, {"dp_applied": False}
    if rng is None:
        rng = np.random.default_rng(seed if seed is not None else spec.seed)
    spec.validate()
    # 1. Clip (where)
    clipped, did_clip, orig_norm, clipped_norm = clip_update(update, spec.max_grad_norm)
    # 2. Noise (where)
    sigma = spec.sigma
    noised = add_gaussian_noise(clipped, sigma, rng, secure_rng=spec.secure_rng)
    metrics = {
        "dp_applied": True,
        "dp_clip_norm": spec.max_grad_norm,
        "dp_noise_multiplier": spec.noise_multiplier,
        "dp_sigma": sigma,
        "dp_did_clip": did_clip,
        "dp_orig_norm": round(orig_norm, 6),
        "dp_clipped_norm": round(clipped_norm, 6),
        "dp_noised_norm": round(float(np.linalg.norm(noised)), 6),
        "dp_delta": spec.delta,
        "dp_secure_rng": spec.secure_rng,
    }
    return noised, metrics


def compute_rdp_epsilon(
    sigma: float,
    delta: float,
    rounds: int,
    sampling_rate: float = 1.0,
) -> Optional[float]:
    """Simple RDP accountant for Gaussian mechanism (upper-bound estimate).

    For Gaussian with sigma, RDP of order alpha is alpha / (2 sigma^2) per step.
    With subsampling rate q, we use the standard amplification bound (approx).
    Composition over T rounds sums RDP, then convert to (epsilon, delta)-DP via:
      epsilon = min_alpha ( RDP(alpha) + log(1/delta) / (alpha -1) )

    This is a textbook bound and may be loose; we report it as an estimate.
    Returns None if sigma == 0 or delta invalid.
    """
    if sigma <= 0 or delta <= 0 or delta >= 1 or rounds <= 0:
        return None
    # Search alpha in [2, 64] with finer granularity (xhigh)
    best_eps = None
    for alpha in [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]:
        # Per-step RDP
        if sampling_rate >= 1.0:
            rdp_step = alpha / (2 * sigma * sigma)
        else:
            # Approx subsampled RDP (Mironov et al. bound, simplified)
            # Use q^2 alpha / sigma^2 for small q (upper bound)
            q = sampling_rate
            rdp_step = (q * q * alpha) / (2 * sigma * sigma)
        total_rdp = rdp_step * rounds
        eps = total_rdp + math.log(1 / delta) / (alpha - 1)
        if best_eps is None or eps < best_eps:
            best_eps = eps
    return float(best_eps) if best_eps is not None else None


def privacy_report(spec: PrivacySpec, rounds: int, sampling_rate: float = 1.0) -> Dict[str, Any]:
    """Human-readable privacy report for one experiment."""
    if not spec.enabled:
        return {
            "enabled": False,
            "mode": "none",
            "note": "No DP: raw federated updates are transmitted. FL does not automatically provide complete privacy.",
        }
    eps = compute_rdp_epsilon(spec.sigma, spec.delta, rounds, sampling_rate)
    return {
        "enabled": True,
        "mode": "client_update_dp",
        "where_clipping": "client-side, on update vector (params_after - params_before), before noise, L2 norm bounded by max_grad_norm",
        "where_noise": "client-side, after clipping, Gaussian N(0, sigma^2) per coordinate, sigma = noise_multiplier * max_grad_norm",
        "parameters": {
            "max_grad_norm (C)": spec.max_grad_norm,
            "noise_multiplier (sigma/C)": spec.noise_multiplier,
            "sigma": spec.sigma,
            "delta": spec.delta,
            "rounds": rounds,
            "sampling_rate": sampling_rate,
        },
        "epsilon_estimate": eps,
        "epsilon_note": "RDP upper-bound estimate; tight accounting requires full composition and amplification analysis. Larger sigma = stronger privacy, lower utility.",
        "assumptions": [
            "Client-level DP: neighboring datasets differ by one client's full local data.",
            "L2 sensitivity bounded by C after clipping.",
            "Noise is i.i.d. Gaussian per coordinate, independent across clients/rounds.",
            "No additional amplification claimed beyond sampling_rate; secure_rng=False uses numpy Generator.",
            "Cumulative privacy degrades with rounds; reported epsilon is for T rounds under stated sigma/delta.",
        ],
    }
