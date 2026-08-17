"""Federated learning with Flower (Phase 9, Phase 10, Phase 11).

Client-side training/evaluation stays fully local; the server only ever sees
model parameters and metrics. Partitions are CONSUMED from Phase 4 (saved
index files) — no random partitioning happens inside the FL implementation.

Phase 10 adds FedProx (fedavg | fedprox) with identical infrastructure; the
proximal regularizer is applied client-side only.

Phase 11 adds personalized FL (personalized): FedPer-style — clients keep a
personal head, only the shared body is aggregated; the global evaluation uses
a server-side probe head.
"""

from src.federated.fl.dataset import PartitionClientData  # noqa: F401
from src.federated.fl.client import (  # noqa: F401
    FedAvgClient,
    PersonalizedClient,
    build_client_fn,
    build_personalized_client_fn,
)
from src.federated.fl.strategy import (  # noqa: F401
    FedAvgTracked,
    FedProxTracked,
    PersonalizedTracked,
)
from src.federated.fl.server import (  # noqa: F401
    run_fl_experiment,
    run_fedavg_experiment,
)