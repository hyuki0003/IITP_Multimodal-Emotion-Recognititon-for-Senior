# MER based on GCL for Senior

This repository contains the implementation module for **Multimodal Emotion Recognition (MER)** based on **Graph Contrastive Learning (GCL)**.

## 📌 Introduction
The goal of this project is to recognize emotions from multimodal data (Text and Audio). The model utilizes Graph Contrastive Learning to effectively capture the interplay between different modalities within conversational contexts.

## 📂 Dataset
The database consists of **text** and **audio** recordings acquired from scripts designed to evoke specific emotions.

### Target Emotions (7 Classes)
The model classifies input data into one of the following 7 emotion categories:
1. **Joy** (기쁨)
2. **Neutral** (중립)
3. **Anxiety** (불안)
4. **Embarrassment** (당황)
5. **Hurt** (상처)
6. **Sadness** (슬픔)
7. **Anger** (분노)

## ⚙️ Dependencies
This project is built with **Python** and requires the following libraries:

- `torch`
- `pandas`
- `numpy`
- `sklearn`
- `pyyaml`
- `typing`
- `matplotlib`
- `datetime`

### Installation
1. Clone this repository.
2. Install the required packages using `pip`:
```bash
pip install -r requirements.txt
```

### Usage

Train and evaluate the model by executing as
```bash
python train.py --dataset IITP-SMED --cuda_id 0
```

Available --dataset arguments must be one of [IITP-SMED, IITP-SMED-STT]

You can choose a single GPU, and cuda_id is the order of available GPU devices.

IITP-SMED and IITP-SMED-STT are our empirical datasets constructed by taking funds from IITP in South Korea.

See details of AIHUB-SER datasets online available link.
