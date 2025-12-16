"""
Training pipeline for employee churn prediction.

This script:
1. Loads raw data
2. Applies preprocessing
3. Splits data into train/test
4. Trains the model
5. Saves trained artifacts to disk

Run:
    python src/pipeline/train_pipeline.py
"""

import pickle
from sklearn.model_selection import train_test_split

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":

    # Step 1: Load raw data
    ingestion = DataIngestion("data/hr_employee_churn_data.csv")
    df = ingestion.load_data()

    # Step 2: Transform data
    transformer = DataTransformation()
    X, y = transformer.preprocess(df)

    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Step 4: Train model
    trainer = ModelTrainer()
    model = trainer.train(X_train, y_train)

    # Step 5: Save trained model
    with open("artifacts/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Training completed. Model saved to artifacts/model.pkl")
