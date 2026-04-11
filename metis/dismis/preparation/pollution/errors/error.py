import pandas as pd


class DMV:
    def __call__(self, dataset, positions) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement this method.")
