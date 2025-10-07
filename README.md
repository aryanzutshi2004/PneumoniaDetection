# PneumoniaDetection# 🩺 Pneumonia Detection using Deep Learning

A deep learning–based **Pneumonia Detection System** built with **PyTorch**, featuring a **Streamlit frontend** and **FastAPI backend** for seamless web deployment.  
This project leverages **Optuna for hyperparameter optimization** and uses a **WeightedRandomSampler** to tackle dataset imbalance, improving the model’s recall and precision for pneumonia cases.

---

## 📁 Project Structure

PNEUMONIADETECTION

├── artifacts/

│ ├── best_model.pth # Best performing trained model

│ ├── model_2.pth # Alternative model checkpoint


├── fast-api backend/
│ ├── model_helper.py # Model loading and prediction helper

│ ├── server.py # FastAPI backend for inference

│ ├── requirements.txt # Backend dependencies


├── streamlit frontend/

│ ├── app.py # Streamlit user interface

│ ├── requirements.txt # Frontend dependencies


├── .gitattributes

├── .gitignore

├── LICENSE

└── README.md



---

## 🚀 Features

- **Deep Learning (PyTorch):** CNN model trained for pneumonia classification  
- **WeightedRandomSampler:** Balanced training for imbalanced datasets  
- **FastAPI Backend:** Serves the model via REST API endpoints  
- **Streamlit Frontend:** Interactive web app for image upload and prediction  
- **Optuna:** Automated hyperparameter optimization for improved performance  
- **Comprehensive Evaluation:** Reports accuracy, precision, recall, and F1-score  

---

## 📊 Model Performance

| Metric       | Normal | Pneumonia |
|---------------|---------|-----------|
| **Precision** | 0.97    | 0.88      |
| **Recall**    | 0.78    | 0.99      |
| **F1-Score**  | 0.86    | 0.93      |

**Overall Accuracy:** 91%  
**Macro Average F1:** 0.90  

---

## 🧠 Tech Stack

- **Deep Learning:** PyTorch  
- **Optimization:** Optuna  
- **Frontend:** Streamlit  
- **Backend:** FastAPI  
- **Language:** Python  
- **Deployment Ready:** REST API + Web Interface  

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/PneumoniaDetection.git
cd PneumoniaDetection
```

### 2️⃣ Create Virtual Environments
```bash
cd "fast-api backend"
pip install -r requirements.txt
uvicorn server:app --reload
```
### 3️⃣ Access the App
```bash
cd "../streamlit frontend"
pip install -r requirements.txt
streamlit run app.py

```
