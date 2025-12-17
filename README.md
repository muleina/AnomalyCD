<div>
<!-- <a href="https://github.com/muleina/AnomalyCD/actions/workflows/python-package.yml"><img src="https://github.com/muleina/LA3D/actions/workflows/python-package.yml/badge.svg" alt="AnomalyCD CI Python"></a>-->
<a href="https://arxiv.org/abs/2412.11800"><img src="https://img.shields.io/badge/Preprint-aXriv-red" alt="AnomalyCD PAPER"></a>
<a href="https://github.com/muleina/AnomalyCD/blob/main/notebooks/AnomalyCD_on_CMS_HCAL_Data_Outlier_Detection.ipynb"><img src="https://img.shields.io/badge/OnlineAD-Notebook-blue" alt="AnomalyCD Online-AD HCAL Dataset"></a>
<a href="https://github.com/muleina/AnomalyCD/blob/main/notebooks/AnomalyCD_on_CMS_HCAL_Data_Causal_Discovery_TS.ipynb"><img src="https://img.shields.io/badge/AnomalyCD-Notebook-blue" alt="AnomalyCD GCM HCAL Dataset"></a>
<a href="https://cmshcalweb01.cern.ch/desmod/"><img src="https://img.shields.io/badge/Production-DESMOD-green" alt="AnomalyCD Production"></a>

# AnomalyCD
Official implementation of the journal paper on "AnomalyCD: *Scalable Temporal Anomaly Causality Discovery in Large Systems: Achieving Computational Efficiency with Binary Anomaly Flag Data*".

The AnomalyCD integrated pipeline for computationally efficient tools for graphical causal discovery (CD) from large binary anomaly data sets.
 Compared to PCMCI: AnomalyCD achieves 8 to 10X speed boost, 14%-57% spurious graph link reduction, gain F1, FPR, and SHDU by 20.5%, 47%, and 41%, respectively, demonstrating improved GCM accuracy. 

This repo is part of the [DEtector System MOnitoring and Diagnostics (DESMOD)](https://cmshcalweb01.cern.ch/desmod) project, a collaboration between the [CMS Experiment at CERN](https://home.cern) and the [University of Agder](https://www.uia.no), Norway.
The DESMOD aims to develop ML tools for the Hadron Calorimeter (HCAL)-Readout Boxes (RBXes), and it includes Anomaly Detection, Anomaly Prediction, and Root-Cause Analysis across large high-dimensional sensor data.

<img src="./docs/images/phd_desmod_method_diagram_2.png" width="100%"/> 

## Abstract 
Extracting anomaly causality facilitates diagnostics once monitoring systems detect system faults. 
Identifying anomaly causes in large systems involves investigating a broader set of monitoring variables across multiple subsystems. 
However, learning causal graphs comes with a significant computational burden that restrains the applicability of most existing methods in real-time and large-scale deployments.
In addition, modern monitoring applications for large systems often generate large amounts of binary alarm flags, and the distinct characteristics of binary anomaly data---the meaning of state transition and data sparsity---challenge existing causality learning mechanisms.
This study proposes an anomaly causal discovery approach (AnomalyCD), addressing the accuracy and computational challenges of generating graphical causal modeling (GCM) from binary flag data sets. 
The AnomalyCD framework presents several strategies, such as anomaly data-aware causality testing (ANAC), sparse data and prior link assemption compression (SDLH), and edge pruning adjustment approaches. 
We validate the performance of this framework on two datasets: monitoring sensor data of the readout-box system of the Compact Muon Solenoid experiment at CERN, and a public data set for information technology monitoring.
The temporal causal discovery results demonstrate a considerable reduction of computation overhead and a moderate enhancement of accuracy on the binary anomaly data sets. 

## AnomalyCD System Design

<img src="./results/rca__online_rca-rca_online_main_2.jpg" width="800"/>

<!-- <img src="./results/rca__online_rca-online_ad_2.jpg" width="600"/> -->

<!-- START doctoc -->
**Table of Contents**
- [Installation](https://github.com/muleina/AnomalyCD#installation)
- [Usage-Notebook](https://github.com/muleina/AnomalyCD#usage-notebook)
- [Results: CMS-HCAL Monitoring Sensor Dataset](https://github.com/muleina/AnomalyCD#The-CMS-HCAL-Monitoring-Sensor-Dataset)
- [Results: EasyVista Monitoring Public Dataset](https://github.com/muleina/AnomalyCD#EasyVista-Monitoring-Public-Dataset)
- [Computational Cost Analysis](https://github.com/muleina/AnomalyCD#Computational-Cost-Analysis)
- [BibTeX Citation](https://github.com/muleina/AnomalyCD#BibTeX-Citation)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Installation

  The requirements.txt will be added soon!
  
    # Anaconda install options: using yml
    conda env create -f conda_environment.yml

    # Anaconda install options: using txt
    conda create --name anomalycd --file conda_requirements.txt

    # Pip install to existing environment.
    pip install -r pip_requirements.txt

## Usage-Notebook

We provide below notebooks for a step-wise result generation of the AnomalyCD pipeline using different data sources. 

-  [HCAL Temporal Online-AD Notebook](https://github.com/muleina/AnomalyCD/blob/main/notebooks/AnomalyCD_on_CMS_HCAL_Data_Outlier_Detection.ipynb) <a href="https://colab.research.google.com/github/muleina/AnomalyCD/blob/main/notebooks/AnomalyCD_on_CMS_HCAL_Data_Outlier_Detection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Online-AD HCAL Dataset In Colab"></a> 
-  [HCAL Temporal Anomaly-CD Notebook](https://github.com/muleina/AnomalyCD/blob/main/notebooks/AnomalyCD_on_CMS_HCAL_Data_Causal_Discovery_TS.ipynb) <a href="https://colab.research.google.com/github/muleina/AnomalyCD/blob/main/notebooks/AnomalyCD_on_CMS_HCAL_Data_Causal_Discovery_TS.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open AnomalyCD HCAL Dataset In Colab"></a> 
-  [EasyVista Temporal Online-AD Notebook](https://github.com/muleina/AnomalyCD/blob/main/notebooks/AnomalyCD_on_EasyVista_Data_Outlier_Detection.ipynb) <a href="https://colab.research.google.com/github/muleina/AnomalyCD/blob/main/notebooks/AnomalyCD_on_EasyVista_Data_Outlier_Detection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Online-AD EasyVista Dataset In Colab"></a> 
-  [EasyVista Temporal Anomaly-CD Notebook](https://github.com/muleina/AnomalyCD/blob/main/notebooks/AnomalyCD_on_EasyVista_Data_Causal_Discovery_TS.ipynb) <a href="https://colab.research.google.com/github/muleina/AnomalyCD/blob/main/notebooks/AnomalyCD_on_EasyVista_Data_Causal_Discovery_TS.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open AnomalyCD EasyVista Dataset In Colab"></a>

## The CMS-HCAL Monitoring Sensor Dataset

The CMS HCAL is a specialized calorimeter that captures hadronic particles during a collision event in the CMS experiment. 
The sensor dataset (2022) contains 4.8M samples from 12 sensors per readout box (RBX), sourced from 36 RBXes of the HCAL front-end electronics that consist of components responsible for sensing and digitizing optical signals of the collision particles. 

<img src="./results/CMS_HCAL/causal_data/CMS_HCAL_HEP07_od_ts_signal.jpg" width="600"/>

### Online-AD

Anomaly detection using our lightwight outlier detection algorithms for time series data sets. 

<img src="./results/CMS_HCAL/causal_data/CMS_HCAL_od_HEP07_1_marked.jpg" width="600"/>

### AnomalyCD

Anomaly temporal graphical causal modeling (GCM) on the binary anomaly flags generated from the online-AD. 

AnomalyCD: sparse data compression reduces the data from 400K to 900 samples (99.8% reduction)

<img src="./results/CMS_HCAL/causal_data/hcal_data_compression.png" width="600"/>

AnomalyCD: graphical causal modeling

<img src="./results/CMS_HCAL/causal_data/hcal_learned_gcm_before_and_after_pruning.png" width="600"/>

AnomalyCD: Performance evaluation

<img src="./results/CMS_HCAL/causal_data/hcal_ablation_perf.png" width="800"/>

The more results are in the paper!

## EasyVista Monitoring Public Dataset

[EasyVista](https://www.easyvista.com/fr/produits/ev-observe) is a publicly available sensor data from their IT monitoring system and can be downlaoded from [here](https://github.com/ckassaad/EasyRCA).
The dataset consists of 8 time series variables collected .   

<img src="./results/rca__easyrca_ground_truth.jpg" width="440"/> <img src="./results/rca__easyrca_rca_ts_idx_positive_0.05_white_5_prune_tlag_first_directed.jpg" width="350"/>

### AnomalyCD

The sparse data compression reduces the data from 4.3K to 1.9K samples (55% reduction). 

Performance evaluation

<img src="./results/EasyVista/causal_data/EasyVista_ablation_study.png" width="800"/>

<img src="./results/EasyVista/causal_data/EasyVista_compare_with_benchmarks.png" width="600"/>

## Computational Cost Analysis

Using Intel(R) Xeon(R) Platinum 8168 CPU @ 2.70GHz with 64GB RAM. Our AnomalyCD has acheived compared to the PCMCI:
- On the HCAL dataset with 99.8% data and 57% link compression.
- 8 to 10X speed boost with half of the memory requirement on the EasyVista dataset with 55% data and 14% link compression. 

Computational Cost on HCAL dataset

<img src="./results/CMS_HCAL/causal_data/cc_analysis/cc_CMS_HCAL.jpg" alt="Computational Cost on HCAL dataset" title="Computational Cost on HCAL dataset" width="600"/>

Computational Cost on HCAL dataset

<img src="./results/EasyVista/causal_data/cc_analysis/cc_EasyVista.jpg" alt="Computational Cost on EasyVista dataset" title="Computational Cost on EasyVista dataset" width="600"/>

## BibTeX Citation

If you employ any part of the code, please kindly cite the following papers:
```
@article{asres2024anomalycd,
  title={Scalable Temporal Anomaly Causality Discovery in Large Systems: Achieving Computational Efficiency with Binary Anomaly Flag Data},
  author={Asres, Mulugeta Weldezgina and Omlin, Christian Walter and {The CMS-HCAL Collaboration}},
  journal={arXiv preprint arXiv:2412.11800},
  year={2024}
}
```

