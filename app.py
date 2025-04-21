import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import io

# Set page configuration
st.set_page_config(
    page_title="Breast Ultrasound Tumor Classification",
    page_icon="🔬",
    layout="wide"
)

# Title and description
st.title("🔬 Breast Ultrasound Tumor Classification")
st.markdown("""
This application uses deep learning to classify breast ultrasound images into three categories:
- **Benign**: Non-cancerous tumors
- **Malignant**: Cancerous tumors
- **Normal**: Normal breast tissue
""")

# Check if model directory exists and create it if it doesn't
model_dir = './model'
if not os.path.exists(model_dir):
    st.warning(f"Model directory '{model_dir}' does not exist. Please ensure you have a trained model.")
    os.makedirs(model_dir, exist_ok=True)

# Function to load the model
@st.cache_resource
def load_classification_model():
    try:
        model_path = './model/model.keras'
        if os.path.exists(model_path):
            model = load_model(model_path)
            return model
        else:
            st.error(f"Model file not found at {model_path}. Please download the model first.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess the image
def preprocess_image(img, target_size=(256, 256)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Function to make prediction
def predict(model, img_array):
    class_labels = {
        0: 'Benign',
        1: 'Malignant', 
        2: 'Normal'
    }
    
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    
    return predicted_class, confidence, predictions[0]

# Main function
def main():
    # Load model
    model = load_classification_model()
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Classification", "About", "Model Details"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader("Choose a breast ultrasound image...", type=["jpg", "jpeg", "png"])
            
            # Sample images option
            st.markdown("### Or try a sample image:")
            sample_dir = "./Dataset_BUSI_with_GT"
            
            if os.path.exists(sample_dir):
                sample_categories = os.listdir(sample_dir)
                sample_categories = [cat for cat in sample_categories if os.path.isdir(os.path.join(sample_dir, cat))]
                
                if sample_categories:
                    selected_category = st.selectbox("Select category", sample_categories)
                    category_path = os.path.join(sample_dir, selected_category)
                    
                    # Get image files (exclude mask files)
                    sample_images = [f for f in os.listdir(category_path) 
                                    if f.endswith(('.png', '.jpg', '.jpeg')) and not f.endswith('_mask.png')]
                    
                    if sample_images:
                        selected_image = st.selectbox("Select sample image", sample_images)
                        sample_image_path = os.path.join(category_path, selected_image)
                        
                        if st.button("Use this sample"):
                            uploaded_file = open(sample_image_path, 'rb')
        
        with col2:
            st.subheader("Classification Results")
            
            if uploaded_file is not None and model is not None:
                # Display the uploaded image
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image", use_column_width=True)
                
                # Preprocess the image
                img_array = preprocess_image(img)
                
                # Predict
                with st.spinner('Classifying...'):
                    predicted_class, confidence, all_probs = predict(model, img_array)
                
                # Display results with color-coded boxes
                if predicted_class == "Benign":
                    st.success(f"Prediction: **{predicted_class}**")
                    box_color = "green"
                elif predicted_class == "Malignant":
                    st.error(f"Prediction: **{predicted_class}**")
                    box_color = "red"
                else:
                    st.info(f"Prediction: **{predicted_class}**")
                    box_color = "blue"
                
                st.markdown(f"Confidence: **{confidence:.2%}**")
                
                # Display probability distribution
                st.subheader("Prediction Probability Distribution")
                fig, ax = plt.subplots(figsize=(6, 2))
                labels = ['Benign', 'Malignant', 'Normal']
                colors = ['green', 'red', 'blue']
                ax.barh(labels, all_probs, color=colors)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probability')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)
                
                # Display additional information based on prediction
                if predicted_class == "Benign":
                    st.markdown("""
                    **Benign tumors** are non-cancerous growths. They:
                    - Don't invade nearby tissues
                    - Don't spread to other parts of the body
                    - Usually have clear boundaries
                    - Typically don't recur if removed completely
                    """)
                elif predicted_class == "Malignant":
                    st.markdown("""
                    **Malignant tumors** are cancerous. They:
                    - Can invade nearby tissues
                    - May spread to other parts of the body (metastasize)
                    - Often have irregular boundaries
                    - Can recur even after treatment
                    
                    ⚠️ **Important**: This is an AI prediction. Please consult with a healthcare professional for accurate diagnosis and treatment.
                    """)
                elif predicted_class == "Normal":
                    st.markdown("""
                    **Normal tissue** shows no signs of abnormal growth. Regular check-ups are still recommended for preventive care.
                    """)
    
    with tab2:
        st.subheader("About This Project")
        st.markdown("""
        ### Breast Ultrasound Tumor Classification

        This project uses deep learning to classify breast ultrasound images into three categories:
        
        - **Benign Tumors**: Non-cancerous growths
        - **Malignant Tumors**: Cancerous growths
        - **Normal Tissue**: No tumors present
        
        ### Dataset
        
        The dataset used in this project is the Breast Ultrasound Images Dataset (BUSI), which contains ultrasound images of breast cancer cases. The dataset includes:
        
        - 437 benign samples
        - 210 malignant samples
        - 133 normal samples
        
        ### Model Architecture
        
        The classification model uses a DenseNet121 architecture pre-trained on ImageNet as a feature extractor, with additional fully connected layers for classification.
        
        ### Purpose
        
        Early detection of breast cancer can significantly improve treatment outcomes. This tool aims to assist medical professionals by providing an additional screening method.
        
        **Note**: This application is for research and educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
        """)
    
    with tab3:
        st.subheader("Model Architecture and Performance")
        
        st.markdown("""
        ### Model Architecture
        
        This project uses a DenseNet121 model pre-trained on ImageNet, with custom classification layers added:
        
        ```
        Model: Sequential
        _________________________________________________________________
         Layer (type)                Output Shape              Param #   
        =================================================================
         densenet121 (Functional)    (None, 8, 8, 1024)        7,037,504
                                                                 
         flatten (Flatten)           (None, 65536)             0         
                                                                 
         dense (Dense)               (None, 1024)              67,109,888
                                                                 
         dense_1 (Dense)             (None, 1024)              1,049,600 
                                                                 
         dense_2 (Dense)             (None, 512)               524,800   
                                                                 
         dense_3 (Dense)             (None, 128)               65,664    
                                                                 
         dense_4 (Dense)             (None, 3)                 387       
                                                                 
        =================================================================
        Total params: 75,787,843
        Trainable params: 68,750,339
        Non-trainable params: 7,037,504
        ```
        
        ### Model Performance
        
        The model was trained for 20 epochs with the following final metrics:
        
        - **Training Accuracy**: 99.36%
        - **Validation Accuracy**: 88.19%
        - **Training Loss**: 0.015
        - **Validation Loss**: 0.652
        """)
        
        
        # Add a section for downloading the model
        st.subheader("Download Pre-trained Model")
        st.markdown("""
        If you haven't downloaded the model yet, you can get it from the following link:
        
        [Download Pre-trained Model](https://drive.google.com/file/d/12VDSpKWK7em7O3-HTx6qWxw5awQUHlIs/view?usp=sharing)
        
        After downloading, place the model file in the `./model/` directory with the filename `model.keras`.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ using TensorFlow and Streamlit")

if __name__ == "__main__":
    main()