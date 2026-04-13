from typing import List

import pandas as pd


class DMV:
    def __call__(self, dataset, positions) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement this method.")

    def get_column_placeholders(
        self, column_names: List[str], example_values: dict
    ) -> tuple[dict, dict, dict]:
        raise NotImplementedError("Subclasses must implement this method.")
