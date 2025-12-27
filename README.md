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

## 🏗 Model Architecture
> Please replace the path below with your actual image file path.

![Model Architecture](./path/to/your/image.png)

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