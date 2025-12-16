import pandas as pd

class CustomData:
    def __init__(
        self,
        satisfaction_level: float,
        last_evaluation: float,
        number_project: int,
        average_montly_hours: int,
        time_spend_company: int,
        work_accident: int,
        promotion_last_5years: int,
        salary_low: int,
        salary_medium: int
    ):
        self.satisfaction_level = satisfaction_level
        self.last_evaluation = last_evaluation
        self.number_project = number_project
        self.average_montly_hours = average_montly_hours
        self.time_spend_company = time_spend_company
        self.work_accident = work_accident
        self.promotion_last_5years = promotion_last_5years
        self.salary_low = salary_low
        self.salary_medium = salary_medium

    def get_data_as_dataframe(self):
        """
        Converts input data into a pandas DataFrame
        """
        data = {
            "satisfaction_level": [self.satisfaction_level],
            "last_evaluation": [self.last_evaluation],
            "number_project": [self.number_project],
            "average_montly_hours": [self.average_montly_hours],
            "time_spend_company": [self.time_spend_company],
            "work_accident": [self.work_accident],
            "promotion_last_5years": [self.promotion_last_5years],
            "low": [self.salary_low],
            "medium": [self.salary_medium]
        }

        return pd.DataFrame(data)
