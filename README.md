# Vision-and-Language Navigation in Continuous Environments (VLN-CE)

> **Cross-Modal Attention Navigation on Matterport3D · Habitat-Lab**

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Habitat](https://img.shields.io/badge/platform-Habitat--Lab-orange)](https://github.com/facebookresearch/habitat-lab)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Overview

This repository implements a full **Vision-and-Language Navigation (VLN)** pipeline using the [VLN-CE](https://jacobkrantz.github.io/vlnce/) framework built on [Habitat-Lab](https://github.com/facebookresearch/habitat-lab). An agent is trained to follow natural language instructions and navigate through photorealistic 3D indoor environments reconstructed from Matterport3D scans.

The project covers end-to-end work across five tasks:

| Task | Description |
|------|-------------|
| **Task 2** | Habitat & VLN-CE environment setup and simulator verification |
| **Task 3** | CMA (Cross-Modal Attention) model training and evaluation |
| **Task 4** | Generalization & ablation studies (unseen environments, paraphrased instructions, reduced data) |
| **Task 5** | Controlled extension — stop-aware policy improvement |

---

## Model Architecture

The navigation agent uses a **Cross-Modal Attention (CMA)** policy:

```
Language Instruction ──► LSTM Encoder ──► Token Features ──┐
                                                           ├──► Cross-Modal Attention ──► Recurrent Policy ──► Action
RGB-D Observation ────► ResNet Encoder ──► Visual Features ─┘
```

- **Visual encoder**: Convolutional backbone (ResNet) pre-trained on PointGoal navigation
- **Language encoder**: Recurrent encoder producing contextual token embeddings
- **Fusion**: Cross-modal attention fuses instruction tokens with the current visual state
- **Policy**: GRU-based recurrent policy predicting discrete navigation actions (`MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT`, `STOP`)
- **Training**: Supervised imitation learning (DAgger / teacher forcing) on the R2R dataset

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **SR** | Success Rate — fraction of episodes where the agent stops within the success threshold of the goal |
| **SPL** | Success weighted by Path Length — rewards both success and path efficiency |
| **NDTW** | Normalized Dynamic Time Warping — measures trajectory fidelity to the reference path |
| **Distance to Goal** | Geodesic distance remaining at episode end |

---

## Repository Structure

```
vln-habitat/
├── project/                          # 📁 Organized project workspace
│   ├── 00_README_START_HERE/         #    Navigation guide for the workspace
│   ├── 01_Task2_Habitat_Setup/       #    Habitat/VLN-CE setup & launcher scripts
│   ├── 02_Task3_CMA_Model/           #    CMA training, checkpoints, logs, report & slides
│   ├── 03_Task4_Generalization_Ablation/  # Unseen eval, paraphrase, reduced-data, ablations
│   ├── 04_Task5_Controlled_Extension/     # Stop-aware extension & baseline comparison
│   ├── 05_Final_Submission/          #    Final ZIPs, slides, and result bundles
│   ├── 90_Runtime_Dependencies/      #    Links to live repos and datasets
│   └── 99_Archive/                   #    Archived non-submission files
│
├── VLN-CE/                           # VLN-CE codebase (training & evaluation scripts)
│   ├── vlnce_baselines/              #    Model implementations (CMA, Seq2Seq, etc.)
│   ├── habitat_extensions/           #    VLN task and sensor extensions for Habitat
│   ├── run.py                        #    Main entry point for train / eval / inference
│   └── scripts/                      #    Utility and graph generation scripts
│
├── utility_dependencies/             # Bundled Habitat utility folders
│   ├── habitat-lab/                  #    Habitat-Lab source
│   ├── habitat-baselines/            #    Habitat-Baselines source
│   └── data/                         #    Datasets, scene assets, checkpoints
│
├── habitat-lab -> utility_dependencies/habitat-lab        # Compatibility symlink
├── habitat-baselines -> utility_dependencies/habitat-baselines  # Compatibility symlink
├── data -> utility_dependencies/data                      # Compatibility symlink
└── README.md                         # This file
```

> **Note**: The root-level `habitat-lab/`, `habitat-baselines/`, and `data/` entries are **symlinks** into `utility_dependencies/` so that existing training and evaluation scripts continue to work without path changes.

---

## Getting Started

### Prerequisites

- Python 3.9+
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (recommended)
- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) with OpenGL support
- [Matterport3D](https://niessner.github.io/Matterport/) scene dataset (requires access agreement)

### Environment Setup

```bash
# Create and activate conda environment
conda create -n habitat python=3.9 cmake=3.14.0
conda activate habitat

# Install Habitat-Sim
conda install habitat-sim withbullet -c conda-forge -c aihabitat

# Install Habitat-Lab
cd utility_dependencies/habitat-lab
pip install -e .

# Install Habitat-Baselines
cd ../habitat-baselines
pip install -e .

# Install VLN-CE dependencies
cd ../../VLN-CE
pip install -r requirements.txt
```

### Data Setup

1. Download Matterport3D scenes into `data/scene_datasets/mp3d/`
2. Download [R2R_VLNCE_v1-3_preprocessed](https://drive.google.com/file/d/1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr/view) into `data/datasets/R2R_VLNCE_v1-3_preprocessed/`
3. Download [depth encoder weights](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7/habitat_baselines/rl/ddppo) into `data/ddppo-models/`

### Training

```bash
cd VLN-CE
python run.py \
  --exp-config vlnce_baselines/config/r2r_baselines/r2r_cma.yaml \
  --run-type train
```

### Evaluation

```bash
python run.py \
  --exp-config vlnce_baselines/config/r2r_baselines/r2r_cma.yaml \
  --run-type eval
```

---

## Project Tasks — Summary

### Task 2: Environment Setup
Configured Habitat-Lab and VLN-CE to run on the target machine. Verified simulator initialization (`Sim-v0`, `VLN-v0`) and confirmed Matterport3D scene loading. Includes Intel/Mesa-safe launcher scripts for non-NVIDIA setups.

### Task 3: CMA Model Training & Evaluation
Trained the Cross-Modal Attention navigation policy using supervised imitation learning on the R2R dataset. Generated navigation videos showing the agent following natural language instructions through 3D environments. Evaluated on SR, SPL, and distance-to-goal metrics.

### Task 4: Generalization & Ablation Studies
Evaluated the trained policy across multiple axes:
- **Unseen environments** : performance gap between seen and unseen splits
- **Paraphrased instructions** : robustness to linguistic variation
- **Reduced training data** : data efficiency analysis
- **Ablation study** : component-level contribution analysis

### Task 5: Controlled Extension — Stop-Aware Policy
Implemented a stop-aware improvement to reduce premature stopping and missed goals. The extension is a targeted change to stopping behavior while keeping the rest of the policy identical for controlled comparison. Results include SR, SPL, and stopping behavior analysis versus the baseline.

---

## Final Submission Artifacts

All submission-ready artifacts are located in [`project/05_Final_Submission/`](project/05_Final_Submission/):

| File | Description |
|------|-------------|
| `vln_acm_overleaf.zip` | Complete Overleaf-ready LaTeX report |
| `vln_task3_task4_methodology_slides.pptx` | Methodology presentation slides |
| `task45_results.zip` | Task 4 & 5 result visualizations and tables |
| `task5_regularization_results.zip` | Stop-aware extension comparison artifacts |

---

## Citations

If you use this work, please cite the underlying frameworks:

```bibtex
@inproceedings{krantz_vlnce_2020,
  title     = {Beyond the Nav-Graph: Vision and Language Navigation in Continuous Environments},
  author    = {Jacob Krantz and Erik Wijmans and Arjun Majundar and Dhruv Batra and Stefan Lee},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2020}
}

@inproceedings{habitat19iccv,
  title     = {Habitat: {A} {P}latform for {E}mbodied {AI} {R}esearch},
  author    = {Manolis Savva and Abhishek Kadian and Oleksandr Maksymets and Yili Zhao and Erik Wijmans and Bhavana Jain and Julian Straub and Jia Liu and Vladlen Koltun and Jitendra Malik and Devi Parikh and Dhruv Batra},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2019}
}
```

## License

This project is MIT licensed. See [LICENSE](LICENSE) for details.

Trained models and task datasets are considered data derived from the Matterport3D scene dataset. Matterport3D-based task datasets and trained models are distributed under [Matterport3D Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) and [CC BY-NC-SA 3.0 US](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).
