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

### Create virtual environment
```bash
python -m venv emp
source emp/bin/activate
'''