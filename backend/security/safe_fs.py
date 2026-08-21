"""Safe filesystem access — no arbitrary filesystem access (Phase 20 §8)."""

from __future__ import annotations

from pathlib import Path


ALLOWED_ROOTS = [
    Path("data").resolve(),
    Path("models").resolve(),
    Path("quarantine").resolve(),
    Path("logs").resolve(),
]


def safe_path(requested: str | Path, allowed_roots: list[Path] | None = None) -> Path:
    """Resolve requested path and ensure it is within an allowed root."""
    roots = allowed_roots or ALLOWED_ROOTS
    p = Path(requested).resolve()
    for root in roots:
        try:
            # Python 3.9+: is_relative_to
            if p.is_relative_to(root):
                return p
        except AttributeError:
            try:
                p.relative_to(root)
                return p
            except ValueError:
                continue
        except Exception:
            continue
    raise PermissionError(f"filesystem access denied: {requested} not within allowed roots")
