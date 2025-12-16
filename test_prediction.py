from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.custom_data import CustomData

data = CustomData(
    satisfaction_level=0.2,
    last_evaluation=0.8,
    number_project=3,
    average_montly_hours=160,
    time_spend_company=3,
    work_accident=0,
    promotion_last_5years=0,
    salary_low=1,
    salary_medium=0
)

df = data.get_data_as_dataframe()
pred = PredictPipeline().predict(df)

print("Prediction:", pred)
