import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 # OpenCV for image reading/processing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import albumentations as A # Needed for Resize in val_test_transform

print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# --- Configuration (MUST MATCH TRAINING SCRIPT for minimal_augment model) ---
DATASET_DIR = './Dataset_BUSI_with_GT'
MODEL_DIR = './model_minimal_augment' # Directory where the minimal augment model is saved
MODEL_FILENAME = 'best_model.keras'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
NUM_CHANNELS = 3
BATCH_SIZE = 16 # Should ideally match training, but can be different for evaluation
RANDOM_STATE = 123 # MUST match the random state used for splitting during training

IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)

# --- 1. Load Data Paths and Labels ---
print(f"\n--- Loading Data from {DATASET_DIR} ---")
if not os.path.isdir(DATASET_DIR):
    raise ValueError(f"Dataset directory not found at: {DATASET_DIR}")

data_paths = []
labels = []
class_names_from_folders = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
print(f"Found classes: {class_names_from_folders}")
num_classes = len(class_names_from_folders)

for folder_name in class_names_from_folders:
    folder_path = os.path.join(DATASET_DIR, folder_name)
    files = glob.glob(os.path.join(folder_path, '*.*'))
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    original_images = []
    skipped_masks = 0
    for f in image_files:
        base_name = os.path.basename(f).lower()
        if '_mask' in base_name: # Filter out masks
            skipped_masks += 1
            continue
        original_images.append(f)

    print(f"Class '{folder_name}': Found {len(original_images)} original images (skipped {skipped_masks} masks).")
    for file_path in original_images:
        data_paths.append(file_path)
        labels.append(folder_name)

all_data_df = pd.DataFrame({'Path': data_paths, 'Label': labels})
print(f"\nTotal original images loaded: {len(all_data_df)}")
if len(all_data_df) == 0:
    raise ValueError("No valid images found after filtering.")

# --- 2. Encode Labels and Split Data (Replicating the exact split) ---
print(f"\n--- Splitting Data (using random_state={RANDOM_STATE}) ---")
label_encoder = LabelEncoder()
# Fit on all labels BEFORE splitting to ensure consistent encoding
all_data_df['Label_Encoded'] = label_encoder.fit_transform(all_data_df['Label'])
class_names = list(label_encoder.classes_) # Get ordered class names from encoder
print(f"Class mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# Perform the same splits as in training
train_df, val_test_df = train_test_split(
    all_data_df,
    train_size=0.8,
    shuffle=True,
    random_state=RANDOM_STATE,
    stratify=all_data_df['Label_Encoded']
)
val_df, test_df = train_test_split(
    val_test_df,
    train_size=0.5,
    shuffle=True,
    random_state=RANDOM_STATE,
    stratify=val_test_df['Label_Encoded']
)

print(f"\nIdentified Test Set Size: {len(test_df)} samples")
if len(test_df) == 0:
    raise ValueError("Test set is empty after splitting. Check dataset size and split ratios.")
print("Test Data Distribution:")
print(test_df['Label'].value_counts())

# --- 3. Define Data Generator Class (Needs to be identical to the one used for training/validation) ---
# Includes DenseNet preprocessing
class DataGenerator(keras.utils.Sequence):
    def __init__(self, df, batch_size, image_size, num_classes, augmentations=None, shuffle=False, preprocess_fn=None): # Ensure shuffle=False for test set
        self.df = df.copy().reset_index(drop=True) # Reset index for predictable ordering
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.augmentations = augmentations # Should be val_test_transform for testing
        self.shuffle = shuffle # Should be False for testing
        self.preprocess_fn = preprocess_fn
        self.indices = np.arange(len(self.df))
        # No shuffle on epoch end needed for testing

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]

        X = np.zeros((len(batch_df), *self.image_size, NUM_CHANNELS), dtype=np.float32)
        y = np.zeros((len(batch_df), self.num_classes), dtype=np.float32)

        for i, (idx, row) in enumerate(batch_df.iterrows()):
            img_path = row['Path']
            label_encoded = row['Label_Encoded']

            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}. Filling with zeros.")
                img = np.zeros((*self.image_size, NUM_CHANNELS), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.augmentations:
                augmented = self.augmentations(image=img)
                img = augmented['image']
            if img.shape[0] != self.image_size[0] or img.shape[1] != self.image_size[1]:
                 img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)

            img = img.astype(np.float32)
            if self.preprocess_fn:
                img = self.preprocess_fn(img)
            else: # Fallback (shouldn't happen if preprocess_fn is passed)
                 img = img / 255.0

            X[i,] = img
            y[i,] = to_categorical(label_encoded, num_classes=self.num_classes)

        return X, y

    def on_epoch_end(self):
        # No shuffling needed for test evaluation
        pass

# --- 4. Create Test Data Generator ---
print("\n--- Creating Test Data Generator ---")
# Define the validation/test transform (Resize only) - Must match training setup
val_test_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=cv2.INTER_LINEAR)
])

test_generator = DataGenerator(
    test_df,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    num_classes=num_classes,
    augmentations=val_test_transform, # Apply resize transform
    shuffle=False,                   # IMPORTANT: No shuffling for test set
    preprocess_fn=densenet_preprocess_input # Use the correct preprocessing
)
print(f"Test generator created with {len(test_generator)} batches.")

# --- 5. Load Trained Model ---
print(f"\n--- Loading Model from {MODEL_PATH} ---")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Ensure the model is trained and saved correctly.")

try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
    # Optional: Print model summary
    # model.summary()
except Exception as e:
    raise RuntimeError(f"Failed to load the model. Error: {e}")

# --- 6. Evaluate Model using model.evaluate ---
print("\n--- Evaluating Model (model.evaluate) ---")
# This provides loss and accuracy calculated by Keras during evaluation loop
loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy (from evaluate): {accuracy*100:.2f}%")

# --- 7. Get Predictions for Detailed Metrics ---
print("\n--- Generating Predictions (model.predict) ---")
# Ensure generator is reset if predict is run multiple times (usually not needed if shuffle=False)
# test_generator.on_epoch_end()
predictions_prob = model.predict(test_generator, verbose=1)

# Get predicted class indices
predictions = np.argmax(predictions_prob, axis=1)

# Get true labels (ensure order matches predictions)
# Since shuffle=False and we reset the index, the order in df should match generator output
y_true = test_df['Label_Encoded'].values

# Verification step (optional but recommended)
if len(predictions) != len(y_true):
    print(f"Warning: Number of predictions ({len(predictions)}) does not match number of true labels ({len(y_true)}). Metrics might be inaccurate.")
    # Attempt to get labels directly from generator (can be slower)
    # y_true_from_gen = []
    # for i in range(len(test_generator)):
    #     _, labels_batch = test_generator[i]
    #     y_true_from_gen.extend(np.argmax(labels_batch, axis=1))
    # y_true = np.array(y_true_from_gen)
    # if len(predictions) != len(y_true):
    #      raise ValueError("Prediction and label count mismatch even after checking generator.")

# --- 8. Calculate and Display Detailed Metrics ---
print("\n--- Calculating Detailed Metrics ---")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, predictions, target_names=class_names, digits=2))

# Accuracy Score (should match model.evaluate closely)
acc_score = accuracy_score(y_true, predictions)
print(f"Test Accuracy (from sklearn): {acc_score*100:.2f}%")

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, predictions)
print(cm)

# --- 9. Visualize Confusion Matrix ---
print("\n--- Visualizing Confusion Matrix ---")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Test Set')
plt.tight_layout()

# Save the plot
cm_filename = 'confusion_matrix_test_set.png'
cm_save_path = os.path.join(MODEL_DIR, cm_filename) # Save inside the model's directory
try:
    plt.savefig(cm_save_path)
    print(f"Confusion matrix saved to: {cm_save_path}")
except Exception as e:
    print(f"Could not save confusion matrix plot. Error: {e}")

# Try to show the plot (might fail in some environments)
try:
    plt.show()
except Exception as e:
    print(f"Could not display confusion matrix plot (might be a non-GUI environment). Error: {e}")


print("\n--- Test Script Finished ---")