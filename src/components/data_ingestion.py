import pandas as pd


class DataIngestion:
    """
    Handles loading of raw data from a source (CSV in this case).

    This class is responsible ONLY for data loading.
    No cleaning or transformation should happen here.
    """

    def __init__(self, data_path: str):
        """
        Parameters
        ----------
        data_path : str
            Path to the raw CSV dataset
        """
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        """
        Loads the dataset from disk.

        Returns
        -------
        pd.DataFrame
            Raw dataset loaded from CSV
        """
        return pd.read_csv(self.data_path)
