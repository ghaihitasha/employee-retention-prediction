from xgboost import XGBClassifier


class ModelTrainer:
    """
    Responsible for training the machine learning model.

    Model selection and hyperparameters can later be
    externalized to config files.
    """

    def train(self, X_train, y_train):
        """
        Trains the XGBoost classifier.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix
        y_train : pd.Series
            Training labels

        Returns
        -------
        model : XGBClassifier
            Trained XGBoost model
        """

        model = XGBClassifier(
            objective='binary:logistic',
            learning_rate=0.1,
            max_depth=20,
            n_estimators=50,
            random_state=42
        )

        model.fit(X_train, y_train)
        return model
