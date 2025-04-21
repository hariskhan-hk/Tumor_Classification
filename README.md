# 🔬 Breast Ultrasound Tumor Classification

<div align="center">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Deep_Learning-007ACC?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0yMCAxMy4xOHYtLjA0YzAtLjI5LjEzLS42NyA0LTIuNVYxMGMtMi0xLjg0LTMuODctMi44NC03LjAzLTIuOTQtMy44Mi0uMTMtNy41MiAyLjE2LTguOTcgNC44NS0uMzcuNjgtLjU4IDEuNDMtLjU4IDIuMi4wMSAxLjkzIDEuMTYgMy42MyAyLjk1IDQuNDcuMi4xLjQyLjE2LjY1LjE2LjE1IDAgLjI5LS4wNS40My0uMTUuNjktLjQ3LjQyLTEuNDMuNDItMS40M2wtLjAxLS4wMWMtMS4xLS41My0xLjgzLTEuNjEtMS44My0yLjg1IDAtLjc2LjI3LTEuNDcuNzgtMi4wNCAxLjE0LTEuMjcgMy4xMi0xLjkxIDUuMDktMS45MXYyLjYzYzAgLjEzLjA2LjI1LjE1LjMzLjA5LjA5LjIxLjE0LjM0LjE0LjI4IDAgLjUtLjIyLjUtLjVWOS41YzEuNzQgMCAzLjQxLjU1IDQuNS44NS42OS4yLjg3LjMzIDEuMjUuODMuMi4yMi41Ni43Mi41NiAxLjAweiIvPjwvc3ZnPg==" alt="Deep Learning">
</div>


## 📋 Overview

This deep learning-based image classification project focuses on classifying breast ultrasound images into three categories:

- 🟢 **Benign**: Non-cancerous tumors
- 🔴 **Malignant**: Cancerous tumors
- 🔵 **Normal**: Healthy breast tissue

The application uses advanced convolutional neural networks (CNNs) with transfer learning from pre-trained models to provide accurate classifications that can assist medical professionals in early diagnosis of breast cancer.

## 🧠 Model Architecture

The classification model leverages a DenseNet121 architecture pre-trained on ImageNet as a feature extractor, with custom fully connected layers for classification:

```
DenseNet121 (pretrained) → Flatten → Dense(1024) → Dense(1024) → Dense(512) → Dense(128) → Dense(3)
```

Performance metrics:
- Training Accuracy: 99.36%
- Validation Accuracy: 88.19%

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.9+
- Streamlit
- Other dependencies listed in requirements.txt

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Tumor_Classification.git
   cd Tumor_Classification
   ```

2. Create a conda environment:
   ```bash
   conda create -n tumor_classification python=3.8
   conda activate tumor_classification
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained model:
   [Download Model](https://drive.google.com/file/d/12VDSpKWK7em7O3-HTx6qWxw5awQUHlIs/view?usp=sharing)
   
   Place the downloaded model in the `./model/` directory with the filename `model.keras`.

5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 📊 Dataset

The project uses the Breast Ultrasound Images Dataset (BUSI), containing:
- 437 benign samples
- 210 malignant samples
- 133 normal samples

Each image is labeled as benign, malignant, or normal, and includes corresponding mask files for the regions of interest.

## 📈 Training Process

The model was trained using transfer learning with the following steps:
1. Feature extraction using DenseNet121 pre-trained on ImageNet
2. Addition of custom fully connected layers for classification
3. Training for 20 epochs with Adam optimizer and categorical cross-entropy loss
4. Data augmentation to improve generalization

## 🔧 Usage

1. Launch the Streamlit app
2. Upload a breast ultrasound image or select a sample image
3. The app will display the classification result with confidence scores
4. Review additional information about the predicted class

## ⚠️ Disclaimer

This application is designed for research and educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.

## 🙏 Acknowledgments

- The BUSI dataset creators
- TensorFlow and Keras development teams
- Streamlit for enabling interactive data applications
