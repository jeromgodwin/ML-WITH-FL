"""Resource-aware federated training (Phase 12).

Real-time protection (file monitoring, feature extraction, inference, risk,
notifications, quarantine) is a separate pipeline and NEVER consults these
modules — federated training is the only consumer, so a paused/deferred/
cancelled training run can never block malware detection.

Public API:
    ResourceMonitor      samples CPU/RAM/battery/AC/idle (graceful when
                         a metric is unsupported)
    ResourcePolicy       config-driven permit/pause/cancel decisions
    TrainingController   start/pause/resume/cancel + the training gate
    create_controller_from_config   factory from a ResourceConfig
"""

from fedshield.config import ResourceConfig

from src.endpoint.resource.controls import (
    STATE_CANCELLED,
    STATE_FINISHED,
    STATE_IDLE,
    STATE_PAUSED,
    STATE_STARTED,
    TrainingController,
    create_controller_from_config,
)
from src.endpoint.resource.monitor import ResourceMonitor
from src.endpoint.resource.policy import PolicyDecision, ResourcePolicy

__all__ = [
    "STATE_CANCELLED", "STATE_FINISHED", "STATE_IDLE", "STATE_PAUSED",
    "STATE_STARTED", "TrainingController", "create_controller_from_config",
    "ResourceMonitor", "ResourcePolicy", "PolicyDecision", "ResourceConfig",
]