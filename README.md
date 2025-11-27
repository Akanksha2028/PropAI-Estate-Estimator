# 🏠 PropAI - Intelligent Real Estate Estimator

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Framework-Flask-green?style=for-the-badge&logo=flask&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Model-XGBoost-orange?style=for-the-badge&logo=xgboost&logoColor=white)
![Frontend](https://img.shields.io/badge/Frontend-Bootstrap_5-purple?style=for-the-badge&logo=bootstrap&logoColor=white)

**PropAI** is an advanced Machine Learning web application designed to bring transparency to the rental market. Unlike standard price calculators, PropAI uses an **XGBoost Regressor** trained on over 4,000 verified listings to provide high-accuracy rent estimates. 

Beyond simple prediction, the system acts as an intelligent advisor—analyzing price-per-square-foot to determine if a property is a **"Steal Deal"**, **"Fair Market Value"**, or **"Overpriced"**.

---

## 🚀 Key Features

* **🧠 Intelligent Prediction:** Estimates monthly rent based on City, Area (SqFt), BHK, Bathrooms, Floor Level, and Furnishing Status.
* **⚖️ Market Verdict System:** Automatically categorizes the predicted price into:
    * 💎 **Steal Deal:** Highly affordable / Underpriced.
    * ⚖️ **Fair Market Value:** Standard pricing.
    * ⚠️ **Premium / Expensive:** Above market average.
* **🎨 Professional UI:** A clean, split-screen modern dashboard built with Bootstrap 5 and Glassmorphism effects.
* **⚡ Real-Time Processing:** Instant inference using a pre-trained serialized model pipeline.

---

## 🛠️ Tech Stack used

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Model** | `XGBoost Regressor` | Extreme Gradient Boosting for high-accuracy regression. |
| **Backend** | `Flask` (Python) | Handles API requests and serves the prediction engine. |
| **Frontend** | `HTML5`, `Bootstrap 5` | Responsive, corporate-grade user interface. |
| **Preprocessing**| `Scikit-Learn` | OneHotEncoding for cities & standardization for numericals. |
| **Data Source** | Kaggle | House Rent Prediction Dataset (4,700+ rows). |

---
## 📸 Project Structure

```text
PropAI-Estate-Estimator/
├── dataset/
│   └── House_Rent_Dataset.csv   # Raw Data
├── models/
│   └── rent_model.pkl           # Trained Model File
├── templates/
│   └── index.html               # Main Web Interface
├── app.py                       # Flask Server & Logic
├── train_model.py               # Model Training Script
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```
### 1. Clone the Repository

To download the code, run this command:
```bash
git clone https://github.com/Akanksha2028/PropAI-Estate-Estimator.git
cd PropAI-Estate-Estimator

