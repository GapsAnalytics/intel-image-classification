# 🧠 Intel Image Classification App 🌍

This project uses deep learning to classify natural scenes into six outdoor categories using the **Intel Image Classification** dataset. It includes:

- A training notebook (`notebook.ipynb`)
- A deployed Streamlit app (`streamlit_app.py`)
- A command-line prediction script (`predict.py`)
- A pretrained model (`model.h5`)

---

## 🗂️ Dataset

Intel Image Classification Dataset from [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification), containing over 10,000 images categorized into:

- 🏙️ `buildings`
- 🌳 `forest`
- 🧊 `glacier`
- ⛰️ `mountain`
- 🌊 `sea`
- 🛣️ `street`

---

## 🧪 Workflow Summary

### ✅ Data Preprocessing
- Resized all images to **64×64**
- Normalized pixel values to **[0, 1]**
- Encoded labels based on folder structure
- Split data into **80% training** and **20% testing**

### ✅ Model Training
- Option A: Custom **CNN**
- Option B: **Transfer Learning** with `MobileNetV2`
- Trained with TensorFlow/Keras and evaluated on test data

### ✅ Evaluation
- Accuracy and **Confusion Matrix** using `sklearn`
- Classification report (precision, recall, F1)

### ✅ Deployment
- `streamlit_app.py`: Web app where users can upload an image and see predicted class
- `predict.py`: Command-line interface for batch prediction

---

## 🚀 Run Locally
