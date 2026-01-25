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
```
## 🚀 Installation & Setup

### 1. Prerequisites
Ensure you have the following installed on your local machine:
* **Python 3.8+**
* **Git**
* **Kaggle Account** (Required to download the dataset automatically)

### 2. Clone the Repository
Clone the project to your local machine using the following command:
```bash
git clone https://github.com/echoesinpages/agri-scan-resnet.git
cd agri-scan-resnet
```
### 3. Create a Virtual Environment (Recommended)
It is best practice to use a virtual environment to manage dependencies:

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```
### 4. Install Dependencies
Install all required Python packages listed in the requirements file:
```bash
pip install -r requirements.txt
```
### 5. Dataset Configuration
This project uses the Kaggle API to fetch the dataset.
1.  Log in to your Kaggle account and go to **Settings** > **API** > **Create New Token**.
2.  This will download a `kaggle.json` file.
3.  Place this `kaggle.json` file in the **root directory** of this project (`agri-scan-resnet/`).
4.  The training script will automatically detect it and download the data.

## 🧠 Model Architecture

The system relies on a **Transfer Learning** approach to balance high accuracy with computational efficiency.

* **Base Model:** **ResNet50** (Pre-trained on ImageNet)
    * **Why ResNet?** Unlike sequential models like VGG16, ResNet50 uses **residual skip-connections**. This allows the network to be much deeper (50 layers) while avoiding the "vanishing gradient" problem, capturing more complex features of plant pathology.
    * **State:** Frozen (Non-trainable) to preserve learned feature extractors.

* **Custom Classification Head:**
    1.  **GlobalAveragePooling2D:** Reduces the spatial dimensions of the feature map (7x7x2048) to a single vector (2048). This is more efficient than "Flatten" and reduces the risk of overfitting.
    2.  **Dropout (0.5):** Randomly deactivates neurons during training to enforce robustness and prevent the model from memorizing the training data.
    3.  **Dense Output Layer:** A fully connected layer with **33 units** (one for each plant class) using the **Softmax** activation function to output probability scores.

 ## 🧠 Training the Model

To start the training pipeline using ResNet50:

```bash
python src/train.py
```
**Pipeline Steps:**
1.  **Ingestion:** Downloads "New Plant Diseases Dataset" from Kaggle.
2.  **Preprocessing:** Resizes images to `224x224`, normalizes pixel values, and applies data augmentation.
3.  **Training:** Fine-tunes the ResNet50 model (Frozen base + Custom Head).
4.  **Evaluation:** Outputs accuracy metrics and saves the best model weights to `model.h5`.
---

## 🚢 Deployment (Inference)

The project includes a lightweight **Flask** server to simulate a production environment.

### Run Locally
```bash
python src/app.py
