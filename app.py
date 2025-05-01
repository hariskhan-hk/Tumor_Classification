import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input # Import specific preprocessor
# from tensorflow.keras.preprocessing import image # Less needed now
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2 # Import OpenCV for resizing consistency
# import io # Not strictly needed in this version

# Set page configuration
st.set_page_config(
    page_title="Breast Ultrasound Tumor Classification",
    page_icon="🔬",
    layout="wide"
)

# Title and description
st.title("🔬 Breast Ultrasound Tumor Classification")
st.markdown("""
This application uses a deep learning model (DenseNet121 base) trained with minimal augmentation
(Horizontal Flip) to classify breast ultrasound images into three categories:
- **Benign**: Non-cancerous tumors
- **Malignant**: Cancerous tumors
- **Normal**: Normal breast tissue
""")

# --- Configuration ---
MODEL_DIR = './model_minimal_augment' # Directory for minimal augment model
MODEL_FILENAME = 'best_model.keras'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
TARGET_SIZE = (256, 256) # Should match training image size
SAMPLE_IMAGE_DIR = './Dataset_BUSI_with_GT'
CLASS_LABELS = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}

# Check if model directory exists
if not os.path.exists(MODEL_DIR):
    st.warning(f"Model directory '{MODEL_DIR}' does not exist. Creating it, but ensure the model file is present.")
    os.makedirs(MODEL_DIR, exist_ok=True)

# Function to load the model
@st.cache_resource # Cache the loaded model
def load_classification_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure the trained model exists.")
        st.error("You might need to train the 'minimal_augment' model or place it in the correct directory.")
        return None
    try:
        # Load the model with compile=False if you don't need to continue training or use the optimizer state
        # This can sometimes avoid compatibility issues.
        model = load_model(MODEL_PATH, compile=False)
        # If you need to compile for predictions with metrics (usually not needed just for predict):
        # model.compile(metrics=['accuracy'])
        st.success(f"Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocessing Function using DenseNet's method
def preprocess_image_densenet(pil_image, target_size=(256, 256)):
    try:
        img_array = np.array(pil_image)
        if img_array.shape[-1] == 4: # Handle RGBA
            img_array = img_array[..., :3]
        elif len(img_array.shape) == 2: # Handle Grayscale
             img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        img_resized = cv2.resize(img_array, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        img_float = img_resized.astype(np.float32)
        img_preprocessed = densenet_preprocess_input(img_float)
        img_batch = np.expand_dims(img_preprocessed, axis=0)
        return img_batch
    except Exception as e:
        st.error(f"Error during image preprocessing: {e}")
        return None

# Function to make prediction
def predict(model, img_array):
    try:
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = CLASS_LABELS[predicted_class_index]
        confidence = predictions[0][predicted_class_index]
        return predicted_class, confidence, predictions[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, 0.0, np.zeros(len(CLASS_LABELS))

# Main function
def main():
    model = load_classification_model()
    tab1, tab2, tab3 = st.tabs(["Classification", "About", "Model Details"])

    with tab1: # Classification Tab
        col1, col2 = st.columns([1, 1])
        uploaded_file_buffer = None # Use buffer to handle sample selection overwrite

        with col1: # Upload/Sample Selection
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader("Choose a breast ultrasound image...", type=["jpg", "jpeg", "png"])

            st.markdown("---")
            st.markdown("#### Or try a sample image:")

            if os.path.exists(SAMPLE_IMAGE_DIR):
                try:
                    sample_categories = sorted([d for d in os.listdir(SAMPLE_IMAGE_DIR) if os.path.isdir(os.path.join(SAMPLE_IMAGE_DIR, d))])
                    if not sample_categories: st.warning("No category subdirectories found.")

                    selected_category = st.selectbox("Select category", sample_categories, index=0 if sample_categories else -1)

                    if selected_category:
                        category_path = os.path.join(SAMPLE_IMAGE_DIR, selected_category)
                        # Robust mask filtering
                        sample_images = sorted([
                            f for f in os.listdir(category_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '_mask' not in f.lower()
                        ])
                        if not sample_images: st.warning(f"No valid (non-mask) images found in '{selected_category}'.")

                        selected_image = st.selectbox("Select sample image", sample_images, index=0 if sample_images else -1)

                        if selected_image and st.button("Use this sample"):
                            sample_image_path = os.path.join(category_path, selected_image)
                            try:
                                uploaded_file_buffer = open(sample_image_path, 'rb') # Read into buffer
                                st.success(f"Using sample: {selected_image}")
                            except Exception as e:
                                st.error(f"Error opening sample image: {e}")
                except Exception as e:
                    st.error(f"Error accessing sample image directory: {e}")
            else:
                st.warning(f"Sample image directory not found: {SAMPLE_IMAGE_DIR}")

        # Use the buffer if a sample was selected, otherwise use the uploaded file
        display_file = uploaded_file_buffer if uploaded_file_buffer is not None else uploaded_file

        with col2: # Results Display
            st.subheader("Classification Results")

            if display_file is not None and model is not None:
                try:
                    img = Image.open(display_file)
                    st.image(img, caption="Selected Image", use_column_width=True)

                    img_array = preprocess_image_densenet(img, target_size=TARGET_SIZE)

                    if img_array is not None:
                        with st.spinner('Classifying...'):
                            predicted_class, confidence, all_probs = predict(model, img_array)

                        if predicted_class is not None:
                            st.markdown("---")
                            if predicted_class == "Benign": st.success(f"### Prediction: **{predicted_class}**")
                            elif predicted_class == "Malignant": st.error(f"### Prediction: **{predicted_class}**")
                            else: st.info(f"### Prediction: **{predicted_class}**")

                            st.markdown(f"#### Confidence: **{confidence:.2%}**")

                            st.subheader("Prediction Probability Distribution")
                            fig, ax = plt.subplots(figsize=(6, 2.5))
                            labels = list(CLASS_LABELS.values())
                            colors = ['green', 'red', 'blue']
                            bars = ax.barh(labels, all_probs, color=colors)
                            ax.set_xlim(0, 1); ax.set_xlabel('Probability')
                            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                            for bar in bars:
                                width = bar.get_width()
                                label_text = f'{width:.1%}' # Format percentage
                                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2., label_text , va='center', ha='left', fontsize=9)
                            st.pyplot(fig)

                            st.markdown("---") # Additional info section
                            if predicted_class == "Benign": st.markdown("""**Benign tumors** are non-cancerous... (rest of text)""")
                            elif predicted_class == "Malignant":
                                st.markdown("""**Malignant tumors** are cancerous... (rest of text)""")
                                st.warning("⚠️ **Important**: AI prediction. Consult a healthcare professional.")
                            elif predicted_class == "Normal": st.markdown("""**Normal tissue** shows no signs... (rest of text)""")
                        else: st.error("Prediction failed.")
                    else: st.error("Image preprocessing failed.")
                except Exception as e:
                     st.error(f"Error processing image: {e}")
                finally:
                    # Close the file buffer if it was opened
                    if uploaded_file_buffer is not None:
                        uploaded_file_buffer.close()
            elif model is None:
                st.error("Model is not loaded. Cannot perform classification.")
            else:
                st.info("Upload an image or select a sample to classify.")

    with tab2: # About Tab
        st.subheader("About This Project")
        st.markdown("""
        *   Uses deep learning (DenseNet121 base) for breast ultrasound classification (Benign, Malignant, Normal).
        *   Trained on the BUSI dataset.
        *   This version uses minimal data augmentation (Horizontal Flip only).
        *   **Disclaimer**: Educational/research tool ONLY. Not for medical diagnosis. Consult a professional.
        """)

    with tab3: # Model Details Tab
        st.subheader("Model Architecture and Performance")
        st.markdown("### Model Architecture")
        # Assuming architecture hasn't fundamentally changed
        st.code("""
Model: "Breast_Tumor_Classifier_Minimal_Augment" (or similar)
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 densenet121 (Functional)    (None, 8, 8, 1024)        7,037,504 (Non-trainable)

 flatten (Flatten)           (None, 65536)             0

 dense_1024_1 (Dense)        (None, 1024)              67,109,888

 dropout_1 (Dropout)         (None, 1024)              0

 dense_512 (Dense)           (None, 512)               524,800

 dropout_2 (Dropout)         (None, 512)               0

 dense_128 (Dense)           (None, 128)               65,664

 output (Dense)              (None, 3)                 387
=================================================================
Total params: 74,737,843
Trainable params: 67,700,339
Non-trainable params: 7,037,504
        """, language='text') # Verify param counts if necessary

        st.markdown("### Model Performance (Minimal Augmentation Model)")
        st.info("""
        Metrics obtained from evaluating the **best saved model** (`model_minimal_augment/best_model.keras`) from the corresponding training run:
        """)
        # --- UPDATED METRICS ---
        st.markdown(f"""
        *   **Training Accuracy (on training set)**: **99.68%**
        *   **Training Loss (on training set)**: **{0.009777:.5f}**
        *   **Best Validation Accuracy (on validation set)**: **91.03%**
        *   **Validation Loss (at best accuracy)**: **{0.416385:.5f}**
        *   **Final Test Set Accuracy (on test set)**: **89.74%**
        *   **Final Test Set Loss (on test set)**: **{0.6009:.4f}**
        """)
        # --- Removed the ACTION NEEDED message ---

        st.subheader("Get Model")
        st.markdown(f"This app expects the model file `{MODEL_FILENAME}` to be located in the `{MODEL_DIR}` directory relative to the script.")
        # If you have a specific download link for *this* version of the model, you can add it here.

    st.markdown("---")
    st.markdown("App using TensorFlow, Albumentations (minimal), and Streamlit.")

if __name__ == "__main__":
    main()