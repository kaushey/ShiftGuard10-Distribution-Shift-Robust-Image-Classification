# ShiftGuard10: Distribution-Shift Robust Image Classification

This repository contains the training pipeline and final report for ShiftGuard10, a robust image classifier developed for a 10-class image classification competition. The primary objective is to achieve robust generalization (target F1 > 0.90) under conditions of severe class imbalance (up to a 50:1 ratio) and test-time distribution shifts.



## Overview

Real-world image classification often suffers from systematic differences between training and test distributions. ShiftGuard10 addresses this using a multi-layered robustness strategy centered around a WideResNet architecture, specialized loss functions, and a heavy augmentation pipeline.

### Key Features
* **Architecture:** WideResNet-40-10 (~54M parameters) acting as the primary backbone. *(Note: Development experiments also utilize WideResNet-28-10).*
* **Loss Function:** `BalancedSoftmaxLoss` combined with label smoothing (ε=0.1) to shift decision boundaries toward minority classes without aggressive oversampling.
* **Optimization:** SGD with Nesterov momentum, linear warmup, and cosine annealing over 300 epochs.
* **Stochastic Weight Averaging (SWA):** Activated at 75% of the training cycle (epoch 225) to find a flatter, more generalizable loss landscape.

## Augmentation Pipeline

To explicitly combat the distribution shifts and specific artifacts present in the dataset, the training pipeline heavily relies on mixed augmentations:

| Strategy | Description |
| :--- | :--- |
| **Domain-Specific** | **GreyCutout:** Injects normalized 5-12px grey patches to mimic actual test-set artifacts. |
| **Batch-Level** | CutMix (25%) and MixUp (25%) applied stochastically. |
| **Image-Level** | AutoAugment (CIFAR-10 policy), ColorJitter, RandomErasing, and GaussianNoise. |

## Inference Strategy

Predictions are stabilized using a 3-seed ensemble (Seeds: 23, 71, 314). During inference, each of the three models undergoes **30-round Test-Time Augmentation (TTA)** using the full augmentation transform, plus one clean pass weighted 3x. This results in 90 total forward passes per test image, averaged to produce the final predictions.

## Performance

Validation performance on the 95/5 stratified split (1,470 validation samples):

| Model Seed | Macro F1-Score |
| :--- | :--- |
| Seed 23 | 0.9206 |
| Seed 71 | 0.8931 |
| Seed 314 | 0.9146 |
| **Ensemble Average** | **0.9094** (Validation) |
| **Ensemble Target** | **> 0.9500** (Test Set) |

## Repository Structure

* `SHIFTGUARD.ipynb`: The primary Jupyter Notebook containing the data loading, model definition, training loop, and TTA inference pipeline. 
* `Group8_report.pdf`: The final project report detailing the architectural choices, ablation studies, and mathematical formulations.

## Usage

The pipeline is designed to run in a PyTorch environment with CUDA support. Automatic Mixed Precision (AMP) is enabled by default to optimize VRAM usage on GPUs (e.g., Tesla T4). 

Execute the cells sequentially in `SHIFTGUARD.ipynb`. The script will automatically cache best-epoch weights and output the final `submission.csv` after ensembling the 3 seeds.
