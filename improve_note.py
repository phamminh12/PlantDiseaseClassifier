import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern, hog
from skimage.filters import gabor
import cv2
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

class KaggleOptimizedPlantDiseaseClassifier:
    def __init__(self):
        self.rf_model = None
        self.svm_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.pca = None

    def optimized_feature_extraction(self, image):
        # Resize to 224x224 as per paper
        image = cv2.resize(image, (224, 224))
        # Process: Đưa tất cả ảnh về kích thước 224x224 để đồng nhất

        # Convert color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Process: Ảnh xám dùng để phân tích kết cấu và hình dạng mà không bị ảnh hưởng bởi màu sắc
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Process: hsv dùng để phân tích màu sắc Không gian màu HSV tốt hơn RGB
        # trong việc phát hiện bệnh vì nó tách biệt thông tin màu (H - hue) khỏi độ sáng (V - value), giúp nhận diện vết bệnh (vàng, nâu, đốm) bất kể điều kiện ánh sáng.

        features = []

        # 1. HOG Features - Most important for texture
        try:
            hog_features = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                              cells_per_block=(2, 2), block_norm='L2-Hys',
                              transform_sqrt=True, feature_vector=True)
            # Process: Nhận vào ảnh xám, tính toán hướng của các cạnh trong các ô 16x16 pixel.
            # Nhận diện hình dạng tổng thể của lá và các đường gân lá hoặc viền của vết bệnh.
            features.extend(hog_features)
        except:
            features.extend([0] * 8100)  # HOG feature size

        # 2. LBP Features - Efficient texture analysis
        try:
            lbp = local_binary_pattern(gray, P=16, R=2, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=64, range=(0, 63))
            # Process: Nhận vào ảnh xám, so sánh độ sáng của một pixel với các pixel lân cận để tạo ra một mã nhị phân, sau đó tính biểu đồ tần suất của các mã này.
            # Phân tích độ sần sùi, lốm đốm của bề mặt lá để phát hiện các vết bệnh nhỏ li ti hoặc các vùng hoại tử trên lá.
            # LBP histogram (64 bins) thể hiện phân bố texture trên lá
            lbp_hist = lbp_hist.astype(float)
            lbp_hist /= (lbp_hist.sum() + 1e-8)
            features.extend(lbp_hist)
        except:
            features.extend([0] * 64)

        # 3. Color Histograms - Most discriminative for plant diseases
        try:
            # HSV histograms
            h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])

            # Normalize histograms
            h_hist = cv2.normalize(h_hist, h_hist).flatten()
            s_hist = cv2.normalize(s_hist, s_hist).flatten()
            v_hist = cv2.normalize(v_hist, v_hist).flatten()
            # Process: Nhận vào ảnh HSV, Tính biểu đồ phân bố màu sắc cho 3 kênh: Hue (Màu), Saturation (Độ bão hòa), Value (Độ sáng)
            # Sau đó chuẩn hóa ảnh dùng để phân biệt bệnh cháy lá, vàng lá, nấm mốc, mốc trắng.

            features.extend(h_hist)
            features.extend(s_hist)
            features.extend(v_hist)
        except:
            features.extend([0] * 96)

        # 4. Statistical features
        try:
            stats_features = [
                np.mean(gray), np.std(gray), np.median(gray),
                np.mean(hsv[:,:,0]), np.std(hsv[:,:,0]),
                np.mean(hsv[:,:,1]), np.std(hsv[:,:,1]),
                np.mean(hsv[:,:,2]), np.std(hsv[:,:,2]),
            ]
            # Process: Nhận vào cả ảnh xám và ảnh HSV, tính trung bình (mean), độ lệch chuẩn (std), trung vị (median) của các giá trị pixel.
            # Tóm tắt tổng quát thông tin toàn cục của ảnh
            features.extend(stats_features)
        except:
            features.extend([0] * 9)

        return np.array(features)

    def load_complete_dataset(self, dataset_path):
        """
        Load ALL 20.6K images - perfectly feasible on Kaggle!
        """
        print("🚀 Loading COMPLETE 20.6K PlantVillage Dataset on Kaggle...")

        features_list = []
        labels_list = []
        class_counts = {}
        total_images = 0

        # Get all class directories
        class_dirs = sorted([d for d in os.listdir(dataset_path)
                           if os.path.isdir(os.path.join(dataset_path, d))])

        print(f"📁 Found {len(class_dirs)} classes: {class_dirs}")

        for class_idx, class_dir in enumerate(class_dirs, 1):
            class_path = os.path.join(dataset_path, class_dir)
            image_files = [f for f in os.listdir(class_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # NO LIMITS - process ALL images
            class_counts[class_dir] = len(image_files)
            total_images += len(image_files)

            print(f"\n🔍 Processing {class_idx}/{len(class_dirs)}: {class_dir} ({len(image_files)} images)...")

            for image_file in tqdm(image_files, desc=f"{class_dir[:15]}..."):
                try:
                    image_path = os.path.join(class_path, image_file)

                    # Read image
                    image = cv2.imread(image_path)
                    if image is None:
                        continue

                    # Extract optimized features
                    features = self.optimized_feature_extraction(image)
                    features_list.append(features)
                    labels_list.append(class_dir)

                except Exception as e:
                    continue

        X = np.array(features_list)
        y = np.array(labels_list)

        print(f"\n🎉 SUCCESS! Complete dataset loaded!")
        print(f"📊 Total images processed: {X.shape[0]}")
        print(f"🔧 Total features per image: {X.shape[1]}")
        print(f"🏷️ Classes: {len(class_dirs)}")
        print(f"📈 Memory usage: ~{X.nbytes / (1024**3):.2f} GB")
        print(f"\n📋 Class distribution:")
        for class_name, count in class_counts.items():
            print(f"   - {class_name}: {count} images")

        return X, y

    def apply_pca(self, X, variance_threshold=0.95):
        """
        Smart PCA to reduce dimensionality while preserving variance
        """
        print(f"\n🔧 Applying PCA for dimensionality reduction...")
        print(f"   Original features: {X.shape[1]}")

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Apply PCA to preserve specified variance
        self.pca = PCA(n_components=variance_threshold, random_state=42)
        X_reduced = self.pca.fit_transform(X_scaled)

        print(f"   Reduced features: {X_reduced.shape[1]}")
        print(f"   Variance preserved: {np.sum(self.pca.explained_variance_ratio_):.4f}")

        return X_reduced

    def train_models(self, X_train, y_train):
        """
        Train models with paper-optimized parameters
        """
        print("\n🤖 Training models with optimized parameters...")

        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        # Apply PCA
        X_train_processed = self.apply_pca(X_train)

        # Random Forest - optimized for accuracy
        self.rf_model = RandomForestClassifier(
            n_estimators=200,           # More trees for better accuracy
            max_depth=50,               # Deeper trees
            min_samples_split=2,        # More splits
            min_samples_leaf=1,         # Fine-grained leaves
            max_features='sqrt',        # Feature sampling
            bootstrap=True,
            random_state=42,
            n_jobs=-1,                  # Use all Kaggle cores
            verbose=1
        )

        # SVM - optimized parameters
        self.svm_model = SVC(
            C=10,                       # Less regularization
            gamma='scale',
            kernel='rbf',
            probability=True,
            random_state=42,
            verbose=1
        )

        print("🌳 Training Random Forest...")
        self.rf_model.fit(X_train_processed, y_train_encoded)

        print("🔷 Training SVM...")
        self.svm_model.fit(X_train_processed, y_train_encoded)

        print("✅ Training completed!")

        return y_train_encoded

    def evaluate_models(self, X_test, y_test):
        """
        Comprehensive model evaluation
        """
        print("\n" + "="*70)
        print("📊 COMPREHENSIVE MODEL EVALUATION")
        print("="*70)

        # Encode labels
        y_test_encoded = self.label_encoder.transform(y_test)

        # Apply PCA to test data
        X_test_scaled = self.scaler.transform(X_test)
        X_test_processed = self.pca.transform(X_test_scaled)

        results = {}

        # Random Forest Evaluation
        print("🔍 Evaluating Random Forest...")
        rf_pred = self.rf_model.predict(X_test_processed)
        rf_pred_proba = self.rf_model.predict_proba(X_test_processed)

        results['Random Forest'] = self._calculate_metrics(y_test_encoded, rf_pred, rf_pred_proba)

        # SVM Evaluation
        print("🔍 Evaluating SVM...")
        svm_pred = self.svm_model.predict(X_test_processed)
        svm_pred_proba = self.svm_model.predict_proba(X_test_processed)

        results['SVM'] = self._calculate_metrics(y_test_encoded, svm_pred, svm_pred_proba)

        # Display results
        self._display_results(results)
        self._plot_results(results)

        return results, rf_pred, svm_pred, y_test_encoded

    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate all evaluation metrics"""
        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'AUC-ROC': roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        }

    def _display_results(self, results):
        """Display results in a professional table"""
        print("\n🏆 MODEL PERFORMANCE COMPARISON")
        print("=" * 85)
        print(f"{'Algorithm':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC-ROC':<12}")
        print("=" * 85)

        for model, metrics in results.items():
            print(f"{model:<20} {metrics['Accuracy']:<12.4f} {metrics['Precision']:<12.4f} "
                  f"{metrics['Recall']:<12.4f} {metrics['F1-Score']:<12.4f} {metrics['AUC-ROC']:<12.4f}")

    def _plot_results(self, results):
        """Plot comparison chart"""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        rf_scores = [results['Random Forest'][m] for m in metrics]
        svm_scores = [results['SVM'][m] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 8))
        bars1 = ax.bar(x - width/2, rf_scores, width, label='Random Forest', color='#2E8B57', alpha=0.8)
        bars2 = ax.bar(x + width/2, svm_scores, width, label='SVM', color='#4682B4', alpha=0.8)

        ax.set_xlabel('Evaluation Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Plant Disease Classification Performance\n(20.6K Images - Complete Dataset)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.legend(fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, score in zip(bars1, rf_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        for bar, score in zip(bars2, svm_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()

def main():
    """
    Main execution - optimized for Kaggle with 20.6K images
    """
    print("🌿 KAGGLE OPTIMIZED PLANT DISEASE CLASSIFICATION")
    print("=" * 60)
    print("🚀 Processing ALL 20.6K Images - No Limits!")
    print("📊 Dataset: 15 classes, ~20,600 total images")
    print("🤖 Algorithms: Random Forest & SVM")
    print("⚡ Optimized for Kaggle Resources")
    print("=" * 60)

    # Initialize classifier
    classifier = KaggleOptimizedPlantDiseaseClassifier()

    # Load dataset - UPDATE THIS PATH for your Kaggle dataset
    # dataset_path = "/kaggle/input/plantdisease/PlantVillage"
    # For: custom python environment
    DATASET_KAGGLEHUB_PATH = "emmarex/plantdisease/versions/1"
    dataset_path = Path.home() / Path(".cache/kagglehub/datasets") / DATASET_KAGGLEHUB_PATH / "PlantVillage"

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please update the dataset_path variable")
        return

    # Load COMPLETE dataset
    X, y = classifier.load_complete_dataset(dataset_path)

    if len(X) == 0:
        print("❌ No images loaded. Check dataset path.")
        return

    # Train-test split (70-30 as per paper)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n📦 Data Split Summary:")
    print(f"   Training set: {X_train.shape[0]:,} images")
    print(f"   Testing set:  {X_test.shape[0]:,} images")
    print(f"   Features per image: {X_train.shape[1]}")

    # Train models
    y_train_encoded = classifier.train_models(X_train, y_train)

    # Evaluate models
    results, rf_pred, svm_pred, y_test_encoded = classifier.evaluate_models(X_test, y_test)

    # Detailed reports
    print("\n" + "="*70)
    print("📋 DETAILED CLASSIFICATION REPORTS")
    print("="*70)

    X_test_scaled = classifier.scaler.transform(X_test)
    X_test_processed = classifier.pca.transform(X_test_scaled)

    print("\n🌳 RANDOM FOREST - Detailed Performance:")
    print("-" * 60)
    print(classification_report(y_test_encoded, rf_pred,
                              target_names=classifier.label_encoder.classes_))

    print("\n🔷 SVM - Detailed Performance:")
    print("-" * 60)
    svm_pred_final = classifier.svm_model.predict(X_test_processed)
    print(classification_report(y_test_encoded, svm_pred_final,
                              target_names=classifier.label_encoder.classes_))

    # Final summary
    print("\n" + "="*70)
    print("🎯 FINAL RESULTS SUMMARY")
    print("="*70)

    rf_acc = results['Random Forest']['Accuracy']
    svm_acc = results['SVM']['Accuracy']

    print(f"🌳 Random Forest Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
    print(f"🔷 SVM Accuracy: {svm_acc:.4f} ({svm_acc*100:.2f}%)")

    # Compare with paper
    paper_rf_accuracy = 0.923  # From the paper
    accuracy_gap = rf_acc - paper_rf_accuracy

    print(f"\n📊 Comparison with Paper:")
    print(f"   Paper RF Accuracy: {paper_rf_accuracy:.4f} (92.3%)")
    print(f"   Our RF Accuracy:   {rf_acc:.4f} ({rf_acc*100:.2f}%)")
    print(f"   Difference:       {accuracy_gap:+.4f}")

    if rf_acc >= paper_rf_accuracy:
        print("🎉 EXCELLENT! Matched or exceeded paper performance!")
    elif rf_acc >= 0.85:
        print("✅ VERY GOOD! Close to paper performance!")
    else:
        print("🔧 Good baseline - further optimization possible")

if __name__ == "__main__":
    main()