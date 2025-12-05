# **Skin-Cancer-Using-CNN**
Deep learning-based skin cancer detection system using CNNs, trained on dermoscopic images. Includes complete training pipeline, data preprocessing, Streamlit deployment, and a ready-to-use classification model for benign vs. malignant lesions.

📘 Table of Contents


## **Features**

- Tech Stack

- Installation

- Dataset

- Model Architecture

- Training

- Run the Web App

- Results

- Future Enhancements

- License

## **🧠 About the Project**

Skin cancer is one of the most common and dangerous forms of cancer.
This project uses a Convolutional Neural Network (CNN) to classify skin lesions as:

✔ Benign
❌ Malignant

Built with TensorFlow, Keras, and deployed using Streamlit, this system aims to help early detection through machine learning.

## **✨ Features**

📌 CNN-based image classification

📸 Accurate detection of skin lesions

🧼 Automated preprocessing & augmentation

🌐 Streamlit web interface

📊 Confusion matrix & metrics

📁 Modular and clean codebase

## **🛠️ Tech Stack**
Technology	Purpose
🧠 TensorFlow/Keras	Deep Learning Model
🖼️ OpenCV	Image Loading & Processing
📊 Matplotlib	Visualization
🌐 Streamlit	Frontend Web App
🐍 Python 3.x	Core Language

## **🔧 Installation**
```
git clone https://github.com/<SriAarushAray>/skin-cancer-detection.git
cd skin-cancer-detection
pip install -r requirements.txt
```
## **📥 Dataset (KaggleHub)**
```
import kagglehub
path = kagglehub.dataset_download("ashenafifasilkebede/dataset")
print("Dataset Path:", path)
```
## **🧩 Model Architecture**
```
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(96, 96, 3)),
    MaxPooling2D(2, 2),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
```
## **🏋️ Training the Model**
```
python src/train.py
```
## **🌐 Run Streamlit App**
```
streamlit run app/main.py
```

## **📊 Results**

✔ 90%+ validation accuracy

✔ Strong binary image classification

✔ Works on unseen dermoscopic images

## **🔮 Future Enhancements**

Add transfer learning (EfficientNet / VGG16)

Improve augmentation & balancing

Add Grad-CAM heatmaps for explainability

Deploy to HuggingFace Spaces

## **📜 License**

This project is licensed under the MIT License.
