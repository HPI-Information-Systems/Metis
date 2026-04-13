import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import expon, kstest, norm, skewnorm, uniform

from metis.dismis.detection.detectors.detector import DMVDetector
from metis.dismis.detection.detectors.utils import force_numeric
from metis.dismis.utils.datetime import datetime_to_numeric


class BucketPDFGoF(DMVDetector):
    def __init__(
        self,
        num_buckets: int = 100,
        method: str = "auto",  # "auto", "norm", "skewnorm", "uniform", "expon"
        z_threshold: float = 3.0,
        min_expected: float = 1e-6,
        target_types: List[str] = ["numeric", "date"],
    ):
        """
        Histogram-vs-PDF goodness-of-fit outlier detector.

        - Optionally auto-selects best distribution (skewnorm, uniform, expon).
        - Fit distribution to data and compare observed vs expected bucket counts.
        - Score samples using absolute standardized residual of their bin.
        - Label samples in bins whose |residual| >= z_threshold.

        Args:
            num_buckets: number of equal-width buckets across the data range.
            method: distribution to fit ("auto", "norm", "skewnorm", "uniform", "expon").
            z_threshold: residual magnitude to flag a bucket as anomalous.
            min_expected: floor to avoid division by zero in sparse tails.
        """
        self.num_buckets = num_buckets
        self.method = method
        self.z_threshold = z_threshold
        self.min_expected = min_expected
        self.target_types = target_types
        self._dist = None

    def _get_dist(self):
        if self._dist is not None:
            return self._dist

        self._dist = {
            "norm": norm,
            "skewnorm": skewnorm,
            "uniform": uniform,
            "expon": expon,
        }
        return self._dist

    def _choose_best_fit(self, x, dists):
        """Test which distribution fits best using KS test p-value."""

        best_method = None
        best_pval = -1.0
        best_params = None

        for name, dist in dists.items():
            try:
                params = dist.fit(x)
                stat, pval = kstest(x, name, args=params)
                if pval > best_pval:
                    best_pval = pval
                    best_method = name
                    best_params = params
            except Exception:
                continue

        return best_method, best_params

    def _extract_features(self, column: pd.Series, type: str) -> np.ndarray:

        if type == "date":
            column, _, _ = datetime_to_numeric(column)

        elif type == "numeric":
            column = force_numeric(column)

        else:
            raise ValueError(f"Column {column.name} must be numeric or date.")

        if len(column) == 0 or column.nunique() <= 1:
            return np.array([])
        column = (column - column.min()) / (column.max() - column.min() + 1e-9)
        values = column.to_numpy(dtype=float)
        return values

    def __call__(
        self,
        dataset: pd.DataFrame,
        types: Dict[str, str],
        target_columns: List[str] | None = None,
        embeddings: Dict[str, pd.DataFrame] = {},
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[str]]:
        times: Dict[str, float] = {
            "preprocessing": 0.0,
            "fitting": 0.0,
            "bucketing": 0.0,
            "scoring": 0.0,
        }
        total_start = time.time()
        assessed: List[str] = []

        df_detect = dataset.copy()
        df_score = pd.DataFrame(0.0, index=dataset.index, columns=dataset.columns)
        df_predict = pd.DataFrame(0, index=dataset.index, columns=dataset.columns)

        if target_columns is None:
            target_columns = dataset.columns.tolist()

        dists = self._get_dist()

        for col in target_columns:
            if types.get(col) not in self.target_types:
                continue

            # --- preprocessing ---
            t0 = time.time()
            series = df_detect[col].dropna()
            x_full = self._extract_features(series, types[col])
            if len(x_full) == 0:
                continue
            mask = ~np.isnan(x_full)
            x = x_full[mask]
            idx = series.index[mask]
            n = x.size
            times["preprocessing"] += time.time() - t0
            if n == 0:
                continue

            # --- fit distribution ---
            t1 = time.time()
            if self.method == "auto":
                method, params = self._choose_best_fit(
                    x, {k: dists[k] for k in ["skewnorm", "uniform", "expon"]}
                )
                if method is None:  # fallback
                    method, params = "norm", (np.mean(x), np.std(x, ddof=0) or 1.0)
            else:
                method = self.method
                params = dists[method].fit(x)

            dist = dists[method]
            cdf = lambda z: dist.cdf(z, *(params or []))
            times["fitting"] += time.time() - t1

            # --- buckets / observed vs expected counts ---
            t2 = time.time()
            xmin, xmax = np.min(x), np.max(x)
            if xmax == xmin:  # degenerate
                df_score.loc[idx, col] = 0.0
                df_predict.loc[idx, col] = 0
                continue

            num_buckets = min(
                self.num_buckets, len(np.unique(x)) // 2
            )  # at least 2 samples per bucket

            edges = np.linspace(xmin, xmax, num_buckets + 1)
            bin_ids = np.clip(
                np.digitize(x, edges, right=False) - 1, 0, num_buckets - 1
            )

            obs = np.bincount(bin_ids, minlength=num_buckets)
            cdf_edges = cdf(edges)
            probs = np.maximum(cdf_edges[1:] - cdf_edges[:-1], 0.0)
            exp = np.maximum(n * probs, self.min_expected)
            resid = (obs - exp) / np.sqrt(exp)
            times["bucketing"] += time.time() - t2

            # --- score & predict ---
            t3 = time.time()
            sample_scores = np.abs(resid[bin_ids])
            anomalous_bins = np.where(np.abs(resid) >= self.z_threshold)[0]
            labels = np.isin(bin_ids, anomalous_bins).astype(int)

            df_score.loc[idx, col] = sample_scores
            assessed.append(col)
            df_predict.loc[idx, col] = labels
            times["scoring"] += time.time() - t3

        times["total"] = time.time() - total_start

        return df_score, df_predict.astype(int), times, assessed


class DistributionFitDetector(DMVDetector):
    def __init__(
        self,
        method: str = "skewnorm",
        density_quantile: float = 0.01,
        target_types: List[str] = ["numeric", "date"],
    ):
        """
        Fit a distribution to values and flag points with low probability density.

        Args:
            method (str): "norm" or "skewnorm".
            density_quantile (float): fraction of lowest PDF densities to mark as outliers.
        """
        self.method = method
        self.density_quantile = density_quantile
        self.target_types = target_types

    def _extract_features(self, column: pd.Series, type: str) -> np.ndarray:

        if type == "date":
            column, _, _ = datetime_to_numeric(column)

        elif type == "numeric":
            column = force_numeric(column)

        else:
            raise ValueError(f"Column {column.name} must be numeric or date.")

        column = (column - column.min()) / (column.max() - column.min() + 1e-9)
        values = column.to_numpy(dtype=float)
        return values

    def __call__(
        self,
        dataset: pd.DataFrame,
        types: Dict[str, str],
        target_columns: List[str] | None = None,
        embeddings: Dict[str, pd.DataFrame] = {},
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[str]]:

        times: Dict[str, float] = {
            "preprocessing": 0,
            "fitting": 0,
            "scoring": 0,
        }
        total_starttime = time.time()
        assessed: List[str] = []

        df_detect = dataset.copy()
        df_score = pd.DataFrame(0.0, index=dataset.index, columns=dataset.columns)
        df_predict = pd.DataFrame(0, index=dataset.index, columns=dataset.columns)

        if target_columns is None:
            target_columns = dataset.columns.tolist()

        for target_column in target_columns:
            if types[target_column] not in self.target_types:
                continue

            preprocessing_starttime = time.time()
            values = self._extract_features(
                df_detect[target_column].dropna(), types[target_column]
            )
            if len(values) == 0:
                continue
            if len(np.unique(values)) == 1:
                # All values are identical; cannot fit a distribution
                continue
            target_idx = df_detect[target_column].dropna().index
            target_idx = np.array(list(range(len(values))))[np.isnan(values) == False]
            values = values[np.isnan(values) == False]
            # print(target_idx)
            times["preprocessing"] += time.time() - preprocessing_starttime

            # --- Fit distribution ---
            fitting_start = time.time()
            if self.method == "skewnorm":
                shape, loc, scale = skewnorm.fit(values)

                pdf_values = skewnorm.pdf(values, shape, loc=loc, scale=scale)
            else:  # default: normal
                mu, sigma = np.mean(values), np.std(values)
                from scipy.stats import norm

                pdf_values = norm.pdf(values, loc=mu, scale=sigma)
            times["fitting"] += time.time() - fitting_start

            # --- Scoring ---
            scoring_start = time.time()
            # print(pdf_values)

            # Convert PDF to outlier score using sigmoid on negative log-likelihood
            # Maps unbounded scores to [0,1] with smooth transition
            # Lower PDF → higher negative log → higher outlier score
            scores = -np.log(pdf_values + 1e-12)
            # scores = 1 / (1 + np.exp(-scores + 5))  # sigmoid with center at 5

            threshold = np.nanquantile(scores, 1 - self.density_quantile)
            labels = (scores >= threshold).astype(int)

            df_score.loc[target_idx, target_column] = scores
            assessed.append(target_column)
            df_predict.loc[target_idx, target_column] = labels
            times["scoring"] += time.time() - scoring_start

        times["total"] = time.time() - total_starttime

        return df_score, df_predict.astype(int), times, assessed
