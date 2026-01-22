# agri-scan-resnet
End-to-end Deep Learning pipeline for plant disease classification. Features a ResNet50 architecture trained on 33 classes, wrapped in a scalable deployment pipeline for real-time inference. Built with TensorFlow and Docker.
# 🌿 Agri-Scan ResNet: End-to-End MLOps

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Deployment](https://img.shields.io/badge/Deployment-Docker-2496ED)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Executive Summary

This repository hosts a production-ready **Computer Vision** solution designed for **Precision Agriculture**. It addresses the challenge of automated plant pathology by deploying a Deep Learning model capable of identifying **33 distinct plant leaf diseases** with high accuracy.

Unlike standard experimental notebooks, this project demonstrates a complete **Machine Learning Lifecycle (MLOps)**, encompassing data ingestion, transfer learning with **ResNet50**, model evaluation, and a deployment pipeline for serving predictions via a REST API.

---

## 🏗️ Technical Architecture

The solution is built on a robust tech stack designed for scalability and performance:

* **Core Model:** **ResNet50** (Deep Residual Network) pre-trained on ImageNet.
* **Technique:** Transfer Learning with fine-tuned top layers for domain adaptation.
* **Input Data:** 33 Classes (New Plant Diseases Dataset).
* **Deployment:** Flask API containerized with Docker.

### Why ResNet50?
We utilize **Deep Residual Learning** to overcome the degradation problem found in shallower networks (like VGG16). The "skip connections" in ResNet50 allow gradients to flow through the network more effectively, enabling the model to learn complex feature hierarchies—such as subtle fungal textures or lesion borders—without suffering from the vanishing gradient problem.

---

## 📂 Directory Structure

```bash
agri-scan-resnet/
├── 📂 data/                   # Dataset storage (ignored by git)
│   └── kaggle.json            # API Key (Manual download required)
├── 📂 notebooks/              # Jupyter Notebooks for experimentation
│   └── exploration.ipynb
├── 📂 src/                    # Source code
│   ├── train.py               # Main training pipeline
│   ├── model.py               # ResNet50 architecture definition
│   └── app.py                 # Flask Inference API
├── Dockerfile                 # Docker configuration for deployment
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── LICENSE                    # MIT License

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure you have the following installed on your local machine:
* **Python 3.8+**
* **Git**
* **Kaggle Account** (Required to download the dataset automatically)

### 2. Clone the Repository
Clone the project to your local machine using the following command:
```bash
git clone [https://github.com/echoesinpages/agri-scan-resnet.git](https://github.com/echoesinpages/agri-scan-resnet.git)
cd agri-scan-resnet
