# FedNAS-GradFree – Representative Code & Data Release

This repository accompanies the paper  
**“Federated learning on the analysis of misinformation verification and validation in social media using machine learning”**.


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

## Running the representative algorithm

```bash
# short demo (20 federated rounds on the 50-device inventory)
python code/fednas_gradfree.py --rounds 20 --dataset FakeNewsNet --seed 42
```

The script writes a JSON log containing the selected architecture per device,  
the global Pareto front, and the simulated communication volume.

---

## Data 

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
