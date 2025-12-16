# Employee Retention Prediction – End-to-End ML Project

This repository contains an end-to-end machine learning system that predicts whether an employee is likely to leave an organization based on historical HR data. The project is designed to closely mirror real-world ML workflows, covering everything from data preprocessing and model training to API deployment and Dockerization.

## Project Overview

Employee attrition is a critical challenge for organizations. This project builds a binary classification model that predicts employee churn (leave vs stay) and exposes the model via a REST API for real-time inference.
The project emphasizes:
- Modular and scalable ML pipeline design
- Separation of training and inference logic
- Production-style folder structure
- Reproducibility and deployment readiness

## Dataset

- HR Employee Churn Dataset
- Target variable: `left`  
  - `1` → Employee left  
  - `0` → Employee stayed  


## Machine Learning Workflow

### 1. Data Ingestion
- Loads raw HR employee data
- Splits data into train and test sets
- Saves intermediate datasets as artifacts

### 2. Data Transformation
- Handles missing values
- Encodes categorical variables (e.g., salary)
- Ensures **training and inference feature consistency**
- Saves preprocessing pipeline

### 3. Model Training
- Trains multiple models:
  - Random Forest
  - XGBoost
- Uses **GridSearchCV** for hyperparameter tuning
- Selects best-performing model
- Saves trained model as an artifact

### 4. Model Evaluation
- Classification Report
- Confusion Matrix
- Comparison against a baseline heuristic

## Inference Pipeline

The **prediction pipeline**:
- Loads the trained model and preprocessor
- Validates feature names and schema
- Returns predictions for new employee data

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
## Tech Stack

### Programming & Core Libraries
- **Python 3.9+** – Core programming language
- **Pandas** – Data manipulation and preprocessing
- **NumPy** – Numerical computations

### Machine Learning
- **scikit-learn** – Model training, evaluation, preprocessing pipelines
- **XGBoost / RandomForest (if used)** – Classification model
- **Joblib** – Model serialization

### Model Engineering
- **Custom ML Pipelines** – Modular design for training and inference
- **Feature Engineering** – Encoding, scaling, and transformations
- **Train/Test Split & Evaluation** – Accuracy, precision, recall, F1-score

### Backend & API
- **Flask** – REST API for serving predictions
- **JSON** – Input/output format for prediction requests

### MLOps & Deployment
- **Docker** – Containerized application for consistent deployment
- **Git & GitHub** – Version control and collaboration
- **Virtual Environments (venv)** – Dependency isolation

### Development & Testing
- **Jupyter Notebook** – EDA and experimentation
- **Postman / curl** – API testing
- **VS Code / Terminal** – Development environment


