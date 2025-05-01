# Breast Ultrasound Tumor Classification

<div align="center">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Albumentations-30A46C?style=for-the-badge&logo=data:image/png;base64,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" alt="Albumentations">
  <img src="https://img.shields.io/badge/OpenCV-2775AE?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
</div>

## Overview

This project implements a deep learning model to classify breast ultrasound images into three distinct categories:

-   🟢 **Benign**: Non-cancerous tumors
-   🔴 **Malignant**: Cancerous tumors
-   🔵 **Normal**: Healthy breast tissue

Leveraging a **DenseNet121** architecture pre-trained on ImageNet and incorporating minimal data augmentation, this application provides classifications intended to assist medical professionals in the early detection process of breast cancer.

## Model Architecture

The classification model utilizes a transfer learning approach:

1.  **Base Model**: DenseNet121 pre-trained on ImageNet (feature extraction layers frozen).
2.  **Custom Head**:
    ```
    Flatten → Dense(1024, ReLU) → Dropout(0.4) → Dense(512, ReLU) → Dropout(0.2) → Dense(128, ReLU) → Dense(3, Softmax)
    ```

### Performance Metrics (Minimal Augmentation Model)

Based on evaluation of the best model saved during training:

-   **Training Accuracy**: 99.68%
-   **Validation Accuracy**: 91.03%
-   **Test Accuracy**: **89.74%**
-   **Test Loss**: 0.6009

*(See the project report for detailed per-class metrics and confusion matrix).*

## Getting Started

### Prerequisites

-   Python 3.8+
-   `pip` (Python package installer)
-   TensorFlow (Tested with 2.9.0)
-   Streamlit
-   OpenCV (`opencv-python-headless`)
-   Albumentations (Tested with 1.4.18)
-   Scikit-learn
-   Pandas, NumPy, Matplotlib, Seaborn
-   (Optional) `huggingface-hub` for model download

### Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/[YourUsername]/[YourRepoName].git # <-- UPDATE THIS URL
    cd [YourRepoName] # <-- UPDATE THIS DIRECTORY NAME
    ```

2.  Create and activate a virtual environment (using `venv`):
    ```bash
    # Create a virtual environment named 'venv' (or your preferred name)
    python -m venv venv

    # Activate the virtual environment
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  Install dependencies using `pip` and the provided `requirements.txt`:
    ```bash
    # Ensure your virtual environment is activated before running pip
    pip install -r requirements.txt
    ```
    *(If `requirements.txt` is not provided, create one or install manually: `pip install tensorflow streamlit opencv-python-headless albumentations scikit-learn pandas numpy matplotlib seaborn huggingface-hub`)*

4.  **Download the Pre-trained Model:**
    The application expects the trained model file `best_model.keras` inside the `./model_minimal_augment/` directory.

    *   **Option 1: Place Manually (Recommended if trained locally)**
        Ensure the `model.keras` is placed in the `./model/` directory.

    *   **Option 2: Download from Hugging Face Hub (If Uploaded)**
        ```python
        # Example using huggingface-hub (install first: pip install huggingface-hub)
        from huggingface_hub import hf_hub_download
        import os

        MODEL_SAVE_DIR = './model'
        MODEL_FILENAME = 'model.keras' # Make sure this matches the uploaded filename
        REPO_ID = "chaoder/tumor-classifier" # Make sure this repo contains the *correct* model

        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        hf_hub_download(repo_id=REPO_ID,
                        filename=MODEL_FILENAME,
                        local_dir=MODEL_SAVE_DIR,
                        local_dir_use_symlinks=False)
        print(f"Model downloaded to {os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)}")
        ```

5.  Run the Streamlit app:
    ```bash
    # Make sure your virtual environment (e.g., venv) is activated
    streamlit run app.py
    ```

## Dataset

The project uses the **Breast Ultrasound Images Dataset (BUSI)**.
*(Reference: Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863)*

-   Contains 780 original grayscale images after filtering masks.
-   Distribution: Benign (437), Malignant (210), Normal (133).
-   Dataset available at sources like Kaggle ([Example Link](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)).

## Training Process (Minimal Augmentation Model)

The final model described here was trained using:

1.  **Feature Extraction**: Using DenseNet121 pre-trained on ImageNet with frozen weights.
2.  **Custom Classifier**: Added fully connected layers with Dropout regularization.
3.  **Data Pipeline**:
    *   Custom Keras `Sequence` data generator.
    *   **Preprocessing**: Images resized to 256x256, converted to RGB, and normalized using `tf.keras.applications.densenet.preprocess_input`.
    *   **Augmentation (Training Only)**: Minimal augmentation using `Albumentations` (Resize + HorizontalFlip with p=0.5).
4.  **Training Loop**:
    *   Adam optimizer with a learning rate of 0.0001.
    *   Categorical Crossentropy loss.
    *   Trained for up to 50 epochs with Early Stopping (patience=10 on `val_accuracy`).
    *   Model Checkpointing saved the best model based on `val_accuracy`.

## Usage

1.  Activate your virtual environment (e.g., `source venv/bin/activate`).
2.  Launch the Streamlit app (`streamlit run app.py`).
3.  Use the "Upload Image" button to select a local breast ultrasound image (`.png`, `.jpg`, `.jpeg`).
4.  Alternatively, select a category and sample image from the sidebar and click "Use this sample".
5.  The app displays the selected image and the model's prediction (Benign, Malignant, or Normal) along with a confidence score and probability distribution chart.
6.  Review the informational text provided for the predicted class.

## Disclaimer

This application is intended solely for **educational and research purposes**. It is **not a medical device** and **must not be used as a substitute** for professional medical advice, diagnosis, or treatment. Clinical decisions should **only** be made by qualified healthcare professionals.

## Acknowledgments

-   The creators and providers of the BUSI dataset.
-   The development teams of TensorFlow, Keras, Albumentations, OpenCV, Scikit-learn, and Streamlit.