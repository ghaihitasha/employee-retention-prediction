import pandas as pd


class DataTransformation:
    """
    Handles feature engineering and preprocessing logic.

    This ensures that:
    - Training and inference use the SAME transformations
    - Notebook logic is reusable in production
    """

    def preprocess(self, df: pd.DataFrame):
        """
        Applies all preprocessing steps required before model training.

        Steps performed:
        - Drop unused columns
        - Handle missing values
        - Encode categorical variables
        - Split features and target

        Parameters
        ----------
        df : pd.DataFrame
            Raw input dataframe

        Returns
        -------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        """

        df.rename(columns={"Work_accident": "work_accident"}, inplace=True)

        # Drop identifier column (not useful for prediction)
        df = df.drop(['empid'], axis=1)

        # Fill missing values in satisfaction_level with mean
        df['satisfaction_level'] = df['satisfaction_level'].fillna(
            df['satisfaction_level'].mean()
        )

        # One-hot encode salary column
        salary_dummies = pd.get_dummies(df['salary'], drop_first=True)

        # Combine encoded columns with main dataframe
        df = pd.concat([df.drop('salary', axis=1), salary_dummies], axis=1)

        # Separate features and target
        X = df.drop('left', axis=1)
        y = df['left']

        return X, y
