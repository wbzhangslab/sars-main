# SARS: A Personalization Perspective towards Fair and Robust against



This repository provides the official PyTorch implementation for the following paper:
>**SARS: A Personalization Perspective towards Fair and Robust against**
>
> **Abstract:** *Federated Learning (FL), an emerging distributed machine learning framework that enables each client to collaboratively train
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

## Acknowlegements
Our code is inspired by [DBA](https://github.com/AI-secure/DBA) and [PFLlib](https://github.com/TsingZ0/PFLlib).
