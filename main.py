import kagglehub

# Download version 1
DATASET_KAGGLEHUB_PATH = "emmarex/plantdisease/versions/1"
path = kagglehub.dataset_download(DATASET_KAGGLEHUB_PATH)

print("Path to PlantVillage dataset files:", path)

# Step 1: Import all necessary libraries
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Libraries for image processing and feature extraction
import cv2
from skimage import feature, io
from tqdm import tqdm
import time

# Step 2: Set up paths and parameters

# For: kaggle notebook
# dataset_path = "/kaggle/input/plantdisease/PlantVillage"
# For: custom python environment
dataset_path = Path.home() / Path(".cache/kagglehub/datasets") / DATASET_KAGGLEHUB_PATH / "PlantVillage"

img_size = (128, 128)

# Step 3: Enhanced feature extraction with detailed progress
def extract_features(image_path):
    """
    Loads an image from the given path and extracts a feature vector.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Resize image to a consistent size
        img = cv2.resize(img, img_size)

        # Convert to different color spaces
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Extract HOG features for texture
        hog_features = feature.hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

        # Extract color histogram features from multiple color spaces
        color_hist_features = []
        for channel, color_img in enumerate([img, img_hsv, img_lab]):
            hist = cv2.calcHist([color_img], [channel], None, [256], [0, 256])
            cv2.normalize(hist, hist)
            color_hist_features.extend(hist.flatten())

        # Combine all features into a single vector
        feature_vector = np.hstack([hog_features, color_hist_features])
        return feature_vector
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

# Step 4: Enhanced preprocessing with comprehensive progress tracking
print("🚀 Starting Plant Disease Detection Pipeline...")
print("=" * 60)

# Track overall start time
pipeline_start_time = time.time()

# Phase 1: Data Collection
print("\n📁 PHASE 1: Data Collection")
print("-" * 30)

image_paths = []
labels = []

print("Scanning dataset directory structure...")
for class_name in tqdm(os.listdir(dataset_path), desc="Reading classes"):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, image_name))
            labels.append(class_name)

print(f"✅ Found {len(image_paths)} images across {len(set(labels))} classes")

# Phase 2: Feature Extraction
print("\n🔍 PHASE 2: Feature Extraction")
print("-" * 30)

features_list = []
valid_labels = []
failed_images = 0

print("Extracting features from images...")
for i in tqdm(range(len(image_paths)), desc="Processing images"):
    feat = extract_features(image_paths[i])
    if feat is not None:
        features_list.append(feat)
        valid_labels.append(labels[i])
    else:
        failed_images += 1

X = np.array(features_list)
y = np.array(valid_labels)

print(f"✅ Successfully processed {len(features_list)} images")
if failed_images > 0:
    print(f"⚠️  Failed to process {failed_images} images (will be skipped)")
print(f"📊 Feature matrix shape: {X.shape}")

# Phase 3: Data Preprocessing
print("\n⚙️  PHASE 3: Data Preprocessing")
print("-" * 30)

print("Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Splitting dataset into train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"✅ Training set: {X_train.shape[0]} samples")
print(f"✅ Test set: {X_test.shape[0]} samples")

print("Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"✅ Reduced feature dimensions: {X_train_pca.shape[1]} (from {X_train.shape[1]})")
print(f"📈 Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# Phase 4: Model Training
print("\n🤖 PHASE 4: Model Training")
print("-" * 30)

# All models now use CPU only (GPU parameters removed)
classifiers = {
    "Support Vector Machine": SVC(kernel='rbf', random_state=42, verbose=False),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=0),
    "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "XGBoost": XGBClassifier(random_state=42, n_jobs=-1, verbosity=0)  # GPU parameter removed
}

results = {}
training_times = {}

# Custom progress bar for model training
models_list = list(classifiers.items())
for idx, (name, clf) in enumerate(models_list, 1):
    print(f"\n📊 Training {name} ({idx}/{len(models_list)})...")

    # Track training time for each model
    start_time = time.time()

    # Create a custom progress indicator for training
    with tqdm(total=100, desc=f"Training {name[:15]:<15}") as pbar:
        clf.fit(X_train_pca, y_train)

        # Simulate progress updates (since most scikit-learn models don't have built-in progress)
        for i in range(100):
            pbar.update(1)
            time.sleep(0.01)  # Small delay to make progress visible

    training_time = time.time() - start_time
    training_times[name] = training_time

    # Make predictions
    y_pred = clf.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"✅ {name} trained in {training_time:.2f}s - Accuracy: {acc:.4f}")

# Phase 5: Results and Analysis
print("\n📊 PHASE 5: Results and Analysis")
print("-" * 30)

# Calculate total pipeline time
total_time = time.time() - pipeline_start_time

print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

# Display results in a nice table format
print(f"\n{'Model':<25} {'Accuracy':<10} {'Training Time':<15}")
print("-" * 50)
for model, accuracy in results.items():
    time_str = f"{training_times[model]:.2f}s"
    print(f"{model:<25} {accuracy:<10.4f} {time_str:<15}")

# Find best model
best_model = max(results, key=results.get)
best_accuracy = results[best_model]

print(f"\n🏆 BEST MODEL: {best_model} with {best_accuracy:.4f} accuracy")
print(f"⏱️  Total pipeline execution time: {total_time:.2f} seconds")

# Additional detailed analysis
print("\n" + "=" * 60)
print("DETAILED ANALYSIS")
print("=" * 60)

print(f"\n📈 Dataset Statistics:")
print(f"   - Total images processed: {len(features_list)}")
print(f"   - Number of classes: {len(label_encoder.classes_)}")
print(f"   - Feature dimension (original): {X.shape[1]}")
print(f"   - Feature dimension (after PCA): {X_train_pca.shape[1]}")
print(f"   - Training set size: {X_train_pca.shape[0]}")
print(f"   - Test set size: {X_test_pca.shape[0]}")

print(f"\n⏰ Performance Insights:")
sorted_times = sorted(training_times.items(), key=lambda x: x[1])
print(f"   - Fastest model: {sorted_times[0][0]} ({sorted_times[0][1]:.2f}s)")
print(f"   - Slowest model: {sorted_times[-1][0]} ({sorted_times[-1][1]:.2f}s)")

# Model comparison
print(f"\n🔍 Model Comparison:")
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for i, (model, acc) in enumerate(sorted_results, 1):
    print(f"   {i}. {model}: {acc:.4f}")

print("\n🎯 Pipeline completed successfully!")

import kagglehub
# Download latest version
path = kagglehub.dataset_download("mohitsingh1804/plantvillage")

print("Path to dataset files:", path)
