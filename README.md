# Speech-based MER via GCL for Senior

This repository contains the implementation module for **Speech+Transcript Emotion Recognition** utilized on **Graph Contrastive Learning (GCL)**.

## 📌 Introduction
The goal of this project is to recognize emotions from multimodal data (Text and Audio). The model utilizes Graph Contrastive Learning to effectively capture the interplay between different modalities within conversational contexts.

## 📂 Dataset
The database consists of **text** and **audio** recordings acquired from scripts designed to evoke specific emotions.

### Target Emotions (7 Classes)
The model classifies input data into one of the following 7 emotion categories:
1. **Joy** (기쁨)
2. **Neutral** (중립)
3. **Afraid** (불안)
4. **Surprise** (당황)
5. **Disgust** (상처)
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

Available datasets should be one of [IITP-SMED, IITP-SMED-STT]

IITP-SMED and IITP-SMED-STT are our empirical datasets constructed by taking funds from IITP in South Korea.

Experiment on additional datasets need to be reproduced by owns.
