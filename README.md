# [ACM IMWUT24] SARS: A Personalized Federated Learning Framework Towards Fairness and Robustness against Backdoor Attacks



This repository provides the official PyTorch implementation for the following paper:
>**SARS: A Personalized Federated Learning Framework Towards Fairness and Robustness against Backdoor Attacks**
>

**Abstract:** *Federated Learning (FL), an emerging distributed machine learning framework that enables each client to collaboratively train
a global model by sharing local knowledge without disclosing local private data, is vulnerable to backdoor model poisoning
attacks. By compromising some users, the attacker manipulates their local training process, and uploads malicious gradient
updates to poison the global model, resulting in the poisoned global model behaving abnormally on the sub-tasks specified by
the malicious user. Previous studies have suggested several methods to defend against backdoor attacks. However, existing
FL backdoor defense methods affect the fairness of the FL system, and fair FL performance may not be robust. Motivated
by these concerns, in this paper, we propose Self-Awareness ReviSion (SARS), a personalized FL framework designed to
resist backdoors and ensure the fairness of the FL system. SARS comprises two key modules: Adaptive Feature Extraction
and Knowledge Mapping. In the Adaptive Feature Extraction module, benign users can adaptively extract clean global
knowledge with self-awareness and self-revision of the backdoor knowledge transferred from the global model. Based on the
previous module, users can effectively ensure the correct mapping of clean sample features and labels. Through extensive
experimental results, SARS can defend against backdoor attacks and improve the fairness of the FL system by comparing
several state-of-the-art FL backdoor defenses or fair FL methods, including FedAvg, Ditto, WeakDP, FoolsGold, and FLAME.*

[![SARS]([/img/sars.png](https://github.com/wbzhangslab/sars-main/blob/main/img/SARS.png) "SARS")]

## Installation
This repository is built in PyTorch 1.8.1 and tested on CUDA 12.1. See requirements.txt for the installation of dependencies required to run SARS.

```
pip install -r requirements.txt
```

## Dataset Preparation
Different datasets (including CIFAR-10, CIFAR-100, and EMNIST, which can be downloaded from their official pages) are utilized to complete different attack scenarios.

| Dataset      | Model Architecture |
| ----------- | ----------- |
| CIFAR-10      | ResNet18       |
| CIFAR-100   | ResNet34        |
|EMNIST|LeNet|



## Quick Start
We divide dataset into different clients under the non-i.i.d scenario. (Take CIFAR-10 as an example)
```shell
cd utils

python generate_cifar10.py noniid - dir
```
Save the experiment results
```
mkdir saved_models
```

Evaluate the robustness against backdoor(DBA)
```shell
python main.py --params utils/cifar10_params.yaml 
             \ --ptrain sars 
             \ --aggregation fedavg 
             \ --attack dba
```

## BibTex Citation
```shell
@article{10.1145/3678571,
author = {Zhang, Weibin and Li, Youpeng and An, Lingling and Wan, Bo and Wang, Xuyu},
title = {SARS: A Personalized Federated Learning Framework Towards Fairness and Robustness against Backdoor Attacks},
year = {2024},
issue_date = {November 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {8},
number = {4},
url = {https://doi.org/10.1145/3678571},
doi = {10.1145/3678571},
month = nov,
articleno = {140},
numpages = {24},
keywords = {Attention Distillation, Backdoor Attack, Fairness, Federated Learning}
}
```

## Acknowlegements
Our code is inspired by [DBA](https://github.com/AI-secure/DBA) and [PFLlib](https://github.com/TsingZ0/PFLlib).
