# 🌿 Agri-Scan ResNet: End-to-End MLOps

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Frontend](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)
![Deployment](https://img.shields.io/badge/Deployment-Docker-2496ED)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Executive Summary

This repository hosts a production-ready **Computer Vision** solution designed for **Precision Agriculture**. It addresses the challenge of automated plant pathology by deploying a Deep Learning model capable of identifying **33 distinct plant leaf diseases** with 96% accuracy.

Unlike standard experimental notebooks, this project demonstrates a complete **Machine Learning Lifecycle (MLOps)**, encompassing data ingestion, transfer learning with **ResNet50**, model evaluation, and a deployment pipeline featuring an interactive **Streamlit Dashboard**.

---

## 🏗️ Technical Architecture

The solution is built on a robust tech stack designed for scalability and performance:

* **Core Model:** **ResNet50** (Deep Residual Network) pre-trained on ImageNet.
* **Technique:** Transfer Learning with fine-tuned top layers for domain adaptation.
* **Input Data:** 33 Classes (New Plant Diseases Dataset).
* **Interface:** Interactive Web Dashboard (Streamlit).
* **Containerization:** Docker for consistent deployment.

### Why ResNet50?
We utilize **Deep Residual Learning** to overcome the degradation problem found in shallower networks (like VGG16). The "skip connections" in ResNet50 allow gradients to flow through the network more effectively, enabling the model to learn complex feature hierarchies—such as subtle fungal textures or lesion borders—without suffering from the vanishing gradient problem.

---

## 📂 Directory Structure

```text
agri-scan-resnet/
├── 📂 docker/                 # Deployment configuration
│   └── Dockerfile             # Container instructions
├── 📂 notebooks/              # Research & Experiments
│   └── agri_scan_resnet50.ipynb # Training notebook (96% Accuracy)
├── 📂 src/                    # Source code
│   ├── app.py                 # Streamlit Web Dashboard (Galaxy Theme)
│   ├── train.py               # Training pipeline script
│   └── model.py               # ResNet50 Architecture
├── 📂 data/                   # Raw Dataset (Ignored by Git)
├── .gitignore                 # Files to exclude from Git
├── model.h5                   # The trained AI Brain (See Download Section)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```
## 🚀 Installation & Setup

### 1. Prerequisites
Ensure you have the following installed on your local machine:
* **Python 3.8+**
* **Git**

### 2. Clone the Repository
Clone the project to your local machine using the following command:
```bash
git clone https://github.com/echoesinpages/agri-scan-resnet.git
cd agri-scan-resnet
```
### 3. 📥 Download the AI Model (Critical Step)
The trained model (`model.h5`) is stored in GitHub Releases to keep the repo light.

1.  Go to the **[Releases Page](https://github.com/echoesinpages/agri-scan-resnet/releases)**.
2.  Download `model.h5` from the latest release.
3.  **Place `model.h5` inside the main folder** (next to `requirements.txt`).

### 4. Create a Virtual Environment
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
### 5. Install Dependencies
Install all required Python packages listed in the requirements file:

```bash
pip install -r requirements.txt
```
## 🚢 How to Run the App

### Option A: Run Locally (Streamlit)
Launch the "Galaxy Edition" dashboard instantly:
```bash
streamlit run src/app.py
```
*The app will open in your browser at http://localhost:8501*

### Option B: Run with Docker (Production)
Build and run the containerized version:
```bash
# Build the image
docker build -t agri-scan-app -f docker/Dockerfile .

# Run the container
docker run -p 8501:8501 agri-scan-app
```
## 🧠 Model Training (Optional)
*Only follow these steps if you want to retrain the model from scratch.*

### 1. Dataset Configuration
This project uses the Kaggle API to fetch the dataset.
1.  Log in to your Kaggle account and go to **Settings** > **API** > **Create New Token**.
2.  This will download a `kaggle.json` file.
3.  Place this `kaggle.json` file in the **root directory** (`agri-scan-resnet/`).

### 2. Start Training
```bash
python src/train.py
```
**Pipeline Steps:**
* **Ingestion:** Downloads "New Plant Diseases Dataset" from Kaggle.
* **Preprocessing:** Resizes images to `224x224`, normalizes pixel values, and applies data augmentation.
* **Training:** Fine-tunes the ResNet50 model (Frozen base + Custom Head).
* **Evaluation:** Outputs accuracy metrics and saves the best model weights to `model.h5`.

## 🧠 Model Architecture Details

The system relies on a **Transfer Learning** approach to balance high accuracy with computational efficiency.

* **Base Model:** **ResNet50** (Pre-trained on ImageNet)
    * **State:** Frozen (Non-trainable) to preserve learned feature extractors.

* **Custom Classification Head:**
    1.  **GlobalAveragePooling2D:** Reduces spatial dimensions (7x7x2048 → 2048).
    2.  **Dropout (0.5):** Randomly deactivates neurons to prevent overfitting.
    3.  **Dense Output Layer:** 33 units with **Softmax** activation for probability scores.
