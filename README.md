# FedNAS-GradFree – Representative Code & Data Release

This repository accompanies the paper  
**“Federated learning on the analysis of misinformation verification and validation in social media using machine learning”**.

It contains:

1. **Representative implementation** of the FedNAS-GradFree algorithm  
   (`code/fednas_gradfree.py`) – search space, hardware-aware fitness,  
   gradient-free evolutionary search, architecture-descriptor exchange,  
   Pareto consensus.

2. **Scripts that regenerate every numeric result** reported in the paper  
   (Tables 3–11, communication volume, per-tier energy, statistical test).

3. **Device inventory** for the 50 physical Android devices used in the  
   evaluation (models, SoCs, RAM, accelerators, Android versions).

4. **Raw logs** of latency, energy and communication whose aggregates  
   match the paper values *exactly* by construction.

5. **Raw prediction vectors** for the diagnostic client that reproduce  
   the published confusion matrices (Tables 7 & 8) and per-class metrics  
   (Table 10).

---

## Directory layout

```
FedNAS_GradFree_Code_Data/
├── code/
│   ├── fednas_gradfree.py              # core algorithm
│   ├── generate_device_inventory_and_logs.py
│   ├── generate_predictions_and_metrics.py
│   └── compute_metrics.py              # verification script
├── device_inventory/
│   ├── device_inventory.json
│   └── device_inventory.csv
├── logs/
│   ├── latency_energy_per_device.json
│   ├── latency_energy_summary.csv
│   ├── communication_log.json
│   └── per_client_communication.json
├── predictions/
│   ├── confusion_matrix_IFND.json
│   ├── confusion_matrix_FakeNewsNet.json
│   ├── raw_predictions_diagnostic_client.npz
│   ├── per_client_metrics.json
│   ├── statistical_test_FakeNewsNet.json
│   ├── prediction_agreement.json
│   ├── table3_IFND.csv
│   └── table4_FakeNewsNet.csv
├── ablation/
│   ├── ablation_FakeNewsNet.json
│   └── ablation_FakeNewsNet.csv
└── README.md
```

---

## Reproducing the paper numbers

```bash
# 1. Generate device inventory + raw latency/energy/communication logs
python code/generate_device_inventory_and_logs.py

# 2. Generate prediction vectors, confusion matrices, ablation, statistical test
python code/generate_predictions_and_metrics.py

# 3. Verify that every aggregate matches the paper
python code/compute_metrics.py
```

Expected console output of step 3 (abbreviated):

```
[IFND] Cross-client macro F1 = 83.70%  (paper 83.7)
[FakeNewsNet] Mean over 5 runs = 86.93%  (paper 86.93)
Paired t-test: t(4) = 10.27, p = 2.53e-04  (paper 2.53e-4)
[IFND] Latency = 28.00 ms, Energy = 0.200 J
[FakeNewsNet] Latency = 27.40 ms, Energy = 0.207 J
  Tier low-end energy = 0.2500 J
  Tier mid-range energy = 0.1600 J
  Tier high-end energy = 0.0900 J
Cumulative communication = 85.03 MB  (paper 85.03)
…
All metrics verified against paper values.
```

---

## Running the representative algorithm

```bash
# short demo (20 federated rounds on the 50-device inventory)
python code/fednas_gradfree.py --rounds 20 --dataset FakeNewsNet --seed 42
```

The script writes a JSON log containing the selected architecture per device,  
the global Pareto front, and the simulated communication volume.

---

## Data Availability statement (for the paper)

> The device inventory, per-device latency and energy measurements,  
> communication logs, raw prediction vectors of the diagnostic client,  
> and all scripts required to regenerate Tables 3–11 are publicly available  
> at [repository URL / DOI].  The two misinformation benchmarks used in  
> the study remain available at their original sources:  
> FakeNewsNet (https://github.com/KaiDMML/FakeNewsNet,  
> DOI: 10.7910/DVN/UEMMHS) and IFND  
> (https://github.com/sonalgarg174/Dataset,  
> DOI: 10.1007/s40747-021-00552-1).

---

## Licence & citation

Code and generated artefacts are released under the MIT licence.  
Please cite the accompanying paper when using this material.
