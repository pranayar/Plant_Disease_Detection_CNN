# 🌿 Plant Disease Detection Using 7-Layer CNN

An AI-powered plant disease detection system using a custom **7-layer Convolutional Neural Network (CNN)** built with TensorFlow/Keras.

The system classifies plant leaf images into **38 different plant health and disease categories** and provides prediction confidence. It also includes a Streamlit web application for image upload and real-time camera-based detection.

---

## 📌 Project Overview

Plant diseases can significantly affect agricultural productivity and crop quality. Early identification of diseases can help farmers and agricultural professionals take appropriate action before infections spread.

This project explores the use of **deep learning and computer vision** to automatically identify plant diseases from leaf images.

A custom 7-layer CNN was designed and trained using the **PlantVillage dataset**. The trained model can then be used through either a web-based Streamlit interface or a desktop application.

### Key Features

* 🧠 Custom **7-layer CNN architecture**
* 🌱 Classification of **38 plant disease/health categories**
* 📸 Image-based plant disease detection
* 🎥 Real-time webcam detection
* 📊 Accuracy, precision, recall and F1-score evaluation
* 📈 Training and validation learning curves
* 🔲 Confusion matrix analysis
* 🔍 Per-class accuracy analysis
* 🎯 Prediction confidence analysis
* 🔥 Grad-CAM visualization
* 🧩 CNN feature-map visualization
* 📉 Multi-class ROC curve analysis
* 📦 TensorFlow Lite model export for potential mobile deployment
* 🌐 Streamlit web interface
* 🖥️ Desktop interface using Tkinter

---

## 🏗️ System Architecture

```text
                    Plant Leaf Image
                           │
                           ▼
                  Image Preprocessing
                  224 × 224 × 3 RGB
                           │
                           ▼
                 ┌──────────────────┐
                 │   7-Layer CNN    │
                 │                  │
                 │ Conv + BN + ReLU │
                 │ Conv + BN + ReLU │
                 │       Pool       │
                 │                  │
                 │ Conv + BN + ReLU │
                 │ Conv + BN + ReLU │
                 │       Pool       │
                 │                  │
                 │ Conv + BN + ReLU │
                 │ Conv + BN + ReLU │
                 │       Pool       │
                 │                  │
                 │ Conv + BN + ReLU │
                 └──────────────────┘
                           │
                           ▼
                 Global Average Pooling
                           │
                           ▼
                     Dense Layer
                           │
                           ▼
                    Softmax Output
                           │
                           ▼
                 Disease Prediction
                 + Confidence Score
```

---

## 🧠 CNN Architecture

The model contains **7 convolutional layers** arranged into four convolutional blocks.

| Block     | Convolution Layers | Filters |
| --------- | ------------------ | ------- |
| Block 1   | 2                  | 32      |
| Block 2   | 2                  | 64      |
| Block 3   | 2                  | 128     |
| Block 4   | 1                  | 256     |
| **Total** | **7**              | —       |

The network also uses:

* Batch Normalization
* ReLU activation
* Max Pooling
* Dropout
* L2 regularization
* Global Average Pooling
* Fully connected classification layer
* Softmax output

Input images are resized to:

```text
224 × 224 × 3
```

The model is trained using the **Adam optimizer** with categorical cross-entropy loss.

---

## 📂 Repository Structure

```text
Research-Methadology/
│
├── PlantVillage_DataSet/
│   └── PlantVillage dataset
│
├── Application.py
│   └── Desktop plant disease detection application
│
├── stream.py
│   └── Streamlit web application
│
├── plant_disease_detection_7_layer.py
│   └── Complete CNN training and evaluation pipeline
│
├── Plant_Disease_Detection_7_Layer -nb.ipynb
│   └── Jupyter/Google Colab notebook
│
├── plant_disease_cnn7_best.keras
│   └── Trained CNN model
│
├── Plant detection - demo.mp4
│   └── Application demonstration
│
├── Technical Report.docx
│   └── Research methodology and technical report
│
└── README.md
```

---

## 🌱 Supported Plant Categories

The model supports 38 classes covering several crops, including:

### 🍎 Apple

* Apple Scab
* Black Rot
* Cedar Apple Rust
* Healthy

### 🫐 Blueberry

* Healthy

### 🍒 Cherry

* Powdery Mildew
* Healthy

### 🌽 Corn

* Cercospora Leaf Spot / Gray Leaf Spot
* Common Rust
* Northern Leaf Blight
* Healthy

### 🍇 Grape

* Black Rot
* Black Measles (Esca)
* Leaf Blight
* Healthy

### 🍊 Orange

* Citrus Greening

### 🍑 Peach

* Bacterial Spot
* Healthy

### 🫑 Bell Pepper

* Bacterial Spot
* Healthy

### 🥔 Potato

* Early Blight
* Late Blight
* Healthy

### 🍓 Raspberry

* Healthy

### 🌱 Soybean

* Healthy

### 🎃 Squash

* Powdery Mildew

### 🍓 Strawberry

* Leaf Scorch
* Healthy

### 🍅 Tomato

* Bacterial Spot
* Early Blight
* Late Blight
* Leaf Mold
* Septoria Leaf Spot
* Spider Mites
* Target Spot
* Yellow Leaf Curl Virus
* Mosaic Virus
* Healthy

---

## 📊 Model Training

The training pipeline includes several techniques designed to improve model performance and generalization.

### Image Preprocessing

Input images are:

* Resized to `224 × 224`
* Converted to RGB
* Normalized to `[0, 1]`

### Data Augmentation

Training images use brightness augmentation:

```text
Brightness range: 0.9 – 1.2
```

### Class Imbalance

Class weights are calculated using:

```python
compute_class_weight(
    class_weight="balanced"
)
```

This helps prevent classes with fewer training examples from being underrepresented during training.

### Training Configuration

| Parameter             | Value                     |
| --------------------- | ------------------------- |
| Image Size            | 224 × 224                 |
| Batch Size            | 32                        |
| Initial Learning Rate | 0.001                     |
| Maximum Epochs        | 50                        |
| Optimizer             | Adam                      |
| Loss                  | Categorical Cross-Entropy |
| Dropout               | 0.25 / 0.40               |
| L2 Regularization     | 0.0001                    |
| Random Seed           | 42                        |

Early stopping and learning-rate reduction are used during training, and the best model is saved based on validation loss.

---

## 📈 Model Evaluation

The project includes several evaluation techniques:

### Classification Metrics

* Accuracy
* Precision
* Recall
* F1-score

Both **macro** and **weighted** metrics are calculated.

### Confusion Matrix

A confusion matrix is generated to identify which plant diseases are most frequently confused with one another.

### Per-Class Accuracy

The project calculates validation accuracy separately for each disease class.

### Prediction Confidence

The distribution of prediction confidence is analyzed to understand how certain the model is about its classifications.

### ROC Curves

Multi-class ROC curves and AUC values are generated for selected disease classes.

---

## 🔥 Explainable AI with Grad-CAM

The project also implements **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

Grad-CAM helps visualize which regions of a leaf image contributed most strongly to the model's prediction.

This provides an additional layer of interpretability beyond simply returning a disease label.

```text
Input Leaf
    │
    ▼
CNN Prediction
    │
    ▼
Gradient Analysis
    │
    ▼
Activation Map
    │
    ▼
Highlighted Disease-Relevant Regions
```

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit application called **GreenLeaf AI**.

The application supports two modes:

### 🖼️ Upload Image

Users can upload:

* JPG
* JPEG
* PNG

The application then returns:

```text
Predicted Disease
Confidence Score
```

### 🎥 Real-Time Camera

The application can also access a webcam and perform continuous plant disease predictions from camera frames.

The Streamlit implementation loads the trained `.keras` model and performs inference on 224 × 224 images.

### Run the Streamlit Application

Install the required Python packages and then run:

```bash
streamlit run stream.py
```

Make sure the trained model is located in the project directory:

```text
plant_disease_cnn7_best.keras
```

---

## 🖥️ Desktop Application

A separate desktop interface is provided through:

```text
Application.py
```

The application uses:

* Python
* Tkinter
* ttkbootstrap
* TensorFlow
* OpenCV
* Pillow

The desktop application loads the trained Keras model and provides a graphical interface for plant disease prediction.

> **Note:** `Application.py` currently contains a local Windows path for the model. Before running it on another computer, update `MODEL_PATH` to point to your local `plant_disease_cnn7_best.keras` file.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/pranayar/Research-Methadology.git
cd Research-Methadology
```

### 2. Install Dependencies

A typical environment requires:

```bash
pip install tensorflow numpy pandas matplotlib seaborn pillow opencv-python scikit-learn tabulate streamlit ttkbootstrap
```

### 3. Run the Web Application

```bash
streamlit run stream.py
```

### 4. Train the Model

The complete training pipeline is available in:

```text
plant_disease_detection_7_layer.py
```

The notebook version is also available:

```text
Plant_Disease_Detection_7_Layer -nb.ipynb
```

The training script was originally developed for Google Colab and expects the PlantVillage dataset to be available at the configured data path.

---

## 📦 TensorFlow Lite

The trained Keras model is also converted to TensorFlow Lite:

```text
plant_disease_cnn7.tflite
```

This provides a path toward deploying the model in resource-constrained or mobile environments.

---

## 🎥 Demo

A demonstration video is included in the repository:

```text
Plant detection - demo.mp4
```

It demonstrates the plant disease detection application and prediction workflow.

---

## 📚 Dataset

This project uses the **PlantVillage dataset** for plant disease classification.

The dataset is associated with the following research publication:

> Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using Deep Learning for Image-Based Plant Disease Detection.* Frontiers in Plant Science, 7, 1419.

**DOI:** 10.3389/fpls.2016.01419

The repository currently includes the PlantVillage dataset and a compressed version of it.

---

## 🔬 Research Methodology

The project follows a typical machine-learning research workflow:

```text
Literature Review
       ↓
Dataset Collection
       ↓
Data Exploration
       ↓
Image Preprocessing
       ↓
CNN Architecture Design
       ↓
Model Training
       ↓
Validation
       ↓
Performance Evaluation
       ↓
Explainability Analysis
       ↓
Application Development
       ↓
Model Deployment
```

The project combines both **experimental machine-learning research** and **practical application development**.

---

## 🛠️ Technologies Used

| Technology         | Purpose             |
| ------------------ | ------------------- |
| Python             | Programming         |
| TensorFlow / Keras | Deep Learning       |
| NumPy              | Numerical Computing |
| Pandas             | Data Analysis       |
| OpenCV             | Computer Vision     |
| Pillow             | Image Processing    |
| Scikit-learn       | Model Evaluation    |
| Matplotlib         | Visualization       |
| Seaborn            | Visualization       |
| Streamlit          | Web Application     |
| Tkinter            | Desktop Application |
| ttkbootstrap       | Desktop UI          |
| TensorFlow Lite    | Model Deployment    |

---

## 🎯 Project Objectives

The primary objectives of this research project are:

1. Develop a custom 7-layer CNN for plant disease classification.
2. Train the model using plant leaf images.
3. Evaluate the model using multiple statistical metrics.
4. Investigate model predictions using explainable AI techniques.
5. Build an interactive application for practical disease detection.
6. Explore the possibility of deploying the trained model to lightweight/mobile platforms.

---

## ⚠️ Limitations

This system is intended as a **research and educational project** rather than a replacement for professional agricultural diagnosis.

Model predictions may be affected by:

* Image quality
* Lighting conditions
* Leaf orientation
* Background complexity
* Disease similarity
* Plant varieties not represented in the training dataset
* Real-world environmental conditions

Predictions should therefore be treated as an AI-assisted indication rather than a definitive agricultural diagnosis.

---

## 🔮 Future Improvements

Potential future improvements include:

* [ ] Larger and more diverse training datasets
* [ ] Additional plant species and diseases
* [ ] Improved data augmentation
* [ ] Transfer learning comparison with architectures such as MobileNet, ResNet and EfficientNet
* [ ] Model quantization
* [ ] Mobile application deployment
* [ ] Cloud deployment
* [ ] Improved real-time detection performance
* [ ] Disease severity estimation
* [ ] Treatment/recommendation information
* [ ] Improved explainability and visualization
* [ ] Automated model benchmarking

---

## 👨‍💻 Author

**Pranay Arora**

MSc Computer Science

This project was developed as part of a **Research Methodology** project focusing on deep learning, computer vision and plant disease detection.

---

## 📄 License

This project is intended primarily for educational and research purposes.

Please refer to the original PlantVillage dataset publication and associated licensing/usage terms when redistributing or reusing the dataset.

---

## ⭐ Acknowledgements

Special thanks to the authors of the PlantVillage dataset and the research community working on computer vision and deep learning for agricultural applications.

### Dataset Reference

Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016).

**Using Deep Learning for Image-Based Plant Disease Detection.**

*Frontiers in Plant Science, 7, 1419.*

DOI: `10.3389/fpls.2016.01419`
