# FedShield

**FedShield: Federated Learning-Based Malware Detection Under Non-IID Client Distributions**

Research-oriented federated learning system for static PE malware detection. Simulates multiple
organizations/endpoints where each client owns a local portion of malware/benign data. Raw training
data stays local; clients communicate only model parameters/updates with a federated server.

**Important research principle:** FedShield keeps raw client data local and investigates
privacy-preserving federated training. Federated learning does not automatically guarantee privacy.
If differential privacy or secure aggregation is not implemented in a given experiment, that is an
explicit limitation of that experiment.

## Objective

Compare, under increasing non-IID severity:

1. Centralized learning
2. FedAvg
3. FedProx
4. Personalized Federated Learning

Metrics: accuracy, precision, recall, F1, ROC-AUC, per-client / worst-client / average-client
performance, convergence across rounds, communication cost, training time, privacy/utility trade-off.

## Dataset

[EMBER 2018_2](https://ember.elastic.co/) — 1.1M static PE feature vectors (2381 features),
binary labels (malware/benign). Downloaded to `data/` (gitignored). EMBER has no per-sample
malware-family labels; family-based heterogeneity is simulated via class-label distribution skew
(Dirichlet) and quantity skew, both seeded and reproducible.

## Development phases

1. Project scaffold *(current)*
2. Dataset pipeline
3. Centralized baseline
4. Non-IID client simulation
5. Flower + FedAvg
6. FedProx
7. Personalized FL
8. Evaluation and experiment engine
9. Communication / privacy / security experiments
10. FastAPI backend
11. React dashboard
12. Testing, reproducibility, final documentation

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then download the dataset:

```powershell
python -m scripts.fetch_ember
```
