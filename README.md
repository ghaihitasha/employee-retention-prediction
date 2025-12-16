# Employee Retention Prediction – End-to-End ML Project

An end-to-end Machine Learning project that predicts whether an employee is likely to leave the company based on HR data.  
This project demonstrates **production-grade ML engineering practices**, covering the full ML lifecycle from data ingestion to model deployment.

---

## Project Overview

Employee attrition is a critical business problem. This project builds a classification model that predicts employee churn using features such as satisfaction level, evaluation score, workload, salary, and promotion history.

The solution is designed with modular, scalable, and reusable components suitable for real-world deployment.

---


---

## Dataset

- HR Employee Churn Dataset
- Target variable: `left`  
  - `1` → Employee left  
  - `0` → Employee stayed  

### Key Features
- satisfaction_level  
- last_evaluation  
- number_project  
- average_montly_hours  
- time_spend_company  
- Work_accident  
- promotion_last_5years  
- salary  

---

## Models Used

### Baseline Model
- Rule-based prediction using satisfaction threshold

### Machine Learning Models
- Random Forest Classifier
- XGBoost Classifier (**final model**)

### Model Selection
- GridSearchCV with 5-fold cross-validation
- Evaluation metrics:
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

---

## How to Run Locally (Without Docker)

### 1. Create virtual environment
```bash
python -m venv emp
source emp/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python -m src.pipeline.train_pipeline
```

This generates
```bash
artifacts/model.pkl
artifacts/preprocessor.pkl
```

### 4. Test prediction pipeline
```bash
python test_prediction.py
```

---

### Run the Flask API
```bash
python app.py
```
API will be available at:
```bash
http://127.0.0.1:8000
```

---

### Prediction API Usage

#### Endpoint
```bash
POST /predict
```

#### Headers
```pgsql
Content-Type: application/json
```

#### Request Body
```json
{
  "satisfaction_level": 0.4,
  "last_evaluation": 0.7,
  "number_project": 3,
  "average_montly_hours": 160,
  "time_spend_company": 3,
  "Work_accident": 0,
  "promotion_last_5years": 0,
  "salary": "low"
}
```

#### Response
```json
{
  "prediction": 1,
  "label": "Likely to leave"
}
```

---

## Dockerized Deployment

### 1. Build Docker Image
```bash
docker build -t employee-retention
```

### 2. Run Docker Container
```bash
docker run -p 8000:8000 employee-retention
```

Access the API at:
```bash
http://localhost:8000/predict
```


