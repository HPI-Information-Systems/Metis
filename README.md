# Metis

Metis is a framework to automatically assess the quality of tabular data across multiple data quality dimensions. The Metis DQ framework (this GitHub repo) is part of the Metis project: [www.metisdq.org](https://www.metisdq.org)

![Overview](images/overview.png)

## Installation

Metis requires Python 3.11+.

```
pip install -r requirements.txt
```

The GUI has additional dependencies (Streamlit, Altair):

```
pip install -r gui/requirements.txt
```

## Start the demo and run Metis:

Metis ships three CLI demos plus an interactive GUI demo. Pick by what you
want to see:

| Demo             | Command                              | What it shows                                                                  |
|------------------|--------------------------------------|--------------------------------------------------------------------------------|
| Getting started  | `python -m demo.getting_started`     | Guided tour: hand-picked metrics on the Adult census dataset, including the four accuracy metrics and a reference-based check |
| Full demo        | `python -m demo.run_demo`            | Every registered metric against the messy restaurants demo dataset             |
| Config-file demo | `python -m demo.metric_config_file`  | Minimal example of passing a metric config as a file path                      |
| GUI demo         | `streamlit run gui/app.py`           | Interactive assessment with precomputed results and a temporal comparison (see [GUI](#gui)) |

```
python -m demo.getting_started
```

The getting-started demo loads `data/adult.csv`, runs a hand-picked selection
of metrics (completeness, minimality, validity, and the four accuracy
metrics), and writes the results to the SQLite repository
`dq_repository/demo.db`. The final step loads a second dataset with an
attached reference (`demo/configs/adult_with_reference.json`) to demonstrate
the reference-based `accuracy_semanticReference` metric.

> **Note on the Acc-I-2 reference.** The getting-started demo exercises
> `accuracy_semanticReference` against `data/adult_gold_sample.csv`, which is a
> synthetic stub built by copying the first 100 rows of `adult.csv` and
> manually injecting two mismatches (row 0 `education`, row 5 `workclass`). It
> is not a real gold standard. It exists only to show the metric mechanically
> detecting the known-planted differences. A real Acc-I-2 run requires an
> external authoritative source for the column(s) under inspection.

## Full demo (all metrics)

To run every registered metric against the demo restaurants dataset, use the extended demo.

```
python -m demo.run_demo
```

Results are written to the SQLite repository `dq_repository/demo.db` (table
`dqresults`).

### The demo dataset

The demo uses `data/restaurants.csv` — a small, intentionally messy dataset
(864 rows) derived from a classic dirty-restaurants benchmark used for
duplicate detection. The source columns are `id`, `name`, `address`, `city`,
`phone`, and `type`; most rows appear twice in slightly different forms (mixed
phone separators, abbreviated city names, divergent cuisine labels), which
gives the duplicate-detection and FD-violation metrics natural raw material to
flag.

The committed CSV is built from `data/restaurants_source.csv` by
`gui/scripts/build_demo_dataset.py`, which appends four synthetic columns and
sprinkles deterministic noise:

```
python gui/scripts/build_demo_dataset.py \
	--source data/restaurants_source.csv \
	--output data/restaurants.csv
```

Synthetic columns (seeded; defaults to `--seed 42`):

| Column                | Distribution                                     |
|-----------------------|--------------------------------------------------|
| `avg_rating`          | beta-distributed in `[1.0, 5.0]`, skewed high    |
| `total_reviews_count` | exponential (mean ≈ 60), integer                 |
| `first_review_date`   | uniform in `2010-01-01` … `2022-01-01`           |
| `last_review_date`    | `first_review_date + uniform(30, 1825)` days     |

Injected noise (also seeded):

- ~10% nulls in the four synthetic columns
- ~3%  nulls in (`name`, `address`, `city`, `phone`, `type`)
- ~2% of date pairs are inverted (`last_review_date < first_review_date`)
- ~2% of `avg_rating` values are pushed outside `[1, 5]`
- ~2% of `total_reviews_count` values are made negative

The deliberate violations exist so the rule-based consistency metrics
(`ruleBasedHinrichs`, `ruleBasedPipino`) and the timeliness/range checks have
something to flag. Tweak the constants at the top of
`gui/scripts/build_demo_dataset.py` (or pass a different `--seed`) to
regenerate.

## GUI

Metis includes a Streamlit GUI that walks through a full assessment in four
steps: upload a dataset, select and configure metrics, compute, and explore
the results visually.

![Metis GUI — results page](images/gui_results.png)

### Quick start

```
pip install -r requirements.txt -r gui/requirements.txt
streamlit run gui/app.py
```

The GUI opens with two flows:

- **Own files** — upload a CSV (plus an optional reference CSV for
  reference-based metrics), pick metrics, and compute. Results are persisted
  locally, so previous runs can be reopened and compared over time.
- **Demo** — a bundled restaurants sample with precomputed results for seven
  metrics across three points in time, so the full results page (including
  the temporal comparison chart) works without computing anything. Set the
  environment variable `METIS_DEMO_ONLY=1` to start the GUI in demo-only
  mode.

For the full GUI documentation, including a walkthrough, demo mode
internals, the dataset/result build scripts, and the architecture of
`gui/core/`, `gui/ui/`, and `gui/visualization/` — see
[docs/GUI.md](docs/GUI.md).

## Using Metis as a library

The `DQOrchestrator` is the main entry point: it loads datasets from data
loader configs, runs metrics from the registry, and hands the results to a
writer.

```python
from metis.dq_orchestrator import DQOrchestrator

orchestrator = DQOrchestrator(writer_config_path="configs/writer/sqlite.json")
orchestrator.load(data_loader_configs=["data/restaurants.json"])
orchestrator.assess(
	metrics=["completeness_nullRatio", "minimality_duplicateCount"],
	metric_configs=[None, None],
)
```

`metrics` and `metric_configs` are parallel lists; each config may be a path
to a JSON file, a JSON string, a pre-instantiated config object, or `None`
(see [How to implement new metrics](#how-to-implement-new-metrics)). If no
`writer_config_path` is given, results are printed to the console.

### Data loader configs

Datasets are described by small JSON configs (see `data/*.json`). File paths
are resolved relative to the `data/` directory:

```json
{
	"loader": "CSV",
	"name": "Adult",
	"file_name": "adult.csv",
	"reference_file_name": "adult_gold_sample.csv",
	"nrows": 100
}
```

`reference_file_name` is optional; when set, the orchestrator loads it as the
reference DataFrame and passes it to every metric run on that dataset.
Further optional fields control CSV parsing (`delimiter`, `encoding`,
`header`, `nrows`, `usecols`, `parse_dates`, `decimals`, `thousands`) and
profile imports (`data_profiles`, see [Data Profiling](#data-profiling)). The
full field reference lives in [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

### Writers

Four writers are available, selected via the `writer_name` field of the
writer config:

| Writer     | `writer_name` | Output                                          |
|------------|---------------|--------------------------------------------------|
| Console    | *(default)*   | Prints each result as JSON to stdout             |
| CSV        | `csv`         | Writes results to a CSV file (`path`)            |
| SQLite     | `sqlite`      | Local SQLite database (`db_name`, `table_name`)  |
| PostgreSQL | `postgres`    | PostgreSQL database (`db_user`, `db_pass`, …)    |

Example configs live in `configs/writer/`. For the PostgreSQL writer, a
ready-to-use database is provided via Docker:

```
docker compose -f docker_compose.yaml up -d
```

Writer config details are also covered in
[docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Available metrics

| Dimension    | Metric                          | What it measures                                                        |
|--------------|---------------------------------|--------------------------------------------------------------------------|
| Accuracy     | `accuracy_syntacticDomain`      | Values belong to an allowed domain, by exact match or WordNet (ISO/IEC 25024 Acc-I-1) |
| Accuracy     | `accuracy_semanticReference`    | Cell agreement with a reference/gold-standard dataset (Acc-I-2)          |
| Accuracy     | `accuracy_outlierRisk`          | Risk of statistical outliers per numeric column, inverted (Acc-I-4)      |
| Accuracy     | `accuracy_dataRange`            | Values fall inside expected intervals (Acc-I-7)                          |
| Completeness | `completeness_nullRatio`        | Ratio of non-null cells                                                  |
| Completeness | `completeness_nullAndDMVRatio`  | Nulls plus disguised missing values (via FAHES)                          |
| Consistency  | `consistency_countFDViolations` | Violations of user-declared functional dependencies                      |
| Consistency  | `consistency_ruleBasedHinrichs` | Rule-based consistency score after Hinrichs (attribute and tuple rules)  |
| Consistency  | `consistency_ruleBasedPipino`   | Rule-based consistency score after Pipino (boolean rules)                |
| Correctness  | `correctness_heinrich`          | Cell-wise correctness against a reference dataset after Heinrich         |
| Minimality   | `minimality_duplicateCount`     | Duplicate rows in the dataset                                            |
| Timeliness   | `timeliness_heinrich`           | Decay-based timeliness of date columns after Heinrich                    |
| Validity     | `validity_outOfVocabulary`      | Share of values outside a known vocabulary                               |

## How to implement new metrics

To extend the Metis framework and add new data quality metrics, please check our interface for easy integration.
````python
def assess(self,
			data: pd.DataFrame,
			reference: pd.DataFrame | None = None,
			metric_config: str | MetricConfig | None = None) -> List[DQResult]:
````
Each metric should be a subclass of ```metis.metric.metric.Metric``` and implement the assess method. This method takes three arguments:
- **data: pandas.Dataframe**: The DataFrame that should be assessed by this metric. This is the primary dataset under inspection.
- **reference: Optional[pd.DataFrame]**: An optional, cleaned reference DataFrame that can act as a gold-standard / ground-truth version of the dataset. Metrics that need a clean version of the data (e.g., correctness against a known-good source) should accept and use this DataFrame. If not needed by a metric, `None` is allowed. The orchestrator loads it via the `reference_file_name` field of the data loader config.
- **metric_config: Optional[str | MetricConfig]**: Optional metric-specific configuration. Accepts a path to a `.json` file, a JSON string, or a pre-instantiated config object; an empty string resolves to a config with all defaults. Use this to keep the method signature compact; all metric-specific parameters (thresholds, aggregation options, etc.) can be stored here.

The metric should return a list of ```metis.utils.result.DQResult```. This can be only one object if one value is computed on a table level or mutliple DQResults if for example one result per column is computed.

**Note:** Each metric has to be imported in the *__init__.py* file inside the folder *metric/* so it is recognized by the Metric registry. Registration itself is automatic: `Metric.__init_subclass__` adds every subclass to `Metric.registry`; the import only triggers it.

### Metric naming convention

Metrics are organized by dimension (e.g., `completeness`, `minimality`), where one folder exists for each.
New metrics should follow the naming format: `{DimensionName}_{Technique}`

- **DimensionName**: The quality dimension being measured (e.g., `Completeness`, `Minimality`)
- **Technique**: The calculation or method used (e.g., `NullRatio`, `DuplicateCount`)

Examples: `completeness_nullRatio`, `minimality_duplicateCount`

The file name and class name of each metric should be equal. If a metric has a specific config class, the name of the config class should be `{MetricName}_config` (e.g., `completeness_missingRatio_config`).

- **Granularity**: The level of analysis (e.g., `cell`, `row`, `column`, `table`) should be passed as a parameter through the metric config file if the metric can be applied at different granularity levels.

### Config conventions

These conventions are required for a metric to be picked up correctly by the
GUI catalog (`gui/core/metric_catalog.py`) and rendered with the right
editor and badges.

#### Config file and class

- Config file lives in the same package as its metric and is named
  `{MetricName}_config.py`.
- The config class name equals the file stem (e.g.
  `completeness_nullRatio_config`).
- The class inherits from `metis.metric.config.MetricConfig` (a dataclass
  with a `validate()` hook) and is itself a `@dataclass`.
- Every field should have a default so the GUI can render the metric
  without forcing the user to fill anything in. Use the `aggregation_axis`
  + `aggregate_all` pattern for metrics that can be summarized at multiple
  granularities:

  ```python
  @dataclass
  class completeness_nullRatio_config(MetricConfig):
	  aggregation_axis: Literal["index", "columns", None] = None
	  aggregate_all: bool = False
  ```

#### Three config types

The GUI dispatches to one of three editors based on metadata declared on
the metric class:

| Type             | Marker on metric class                  | Editor                          |
|------------------|-----------------------------------------|---------------------------------|
| Dataclass config | (default — just provide a config class) | `simple_editor`                 |
| Callable rules   | `_gui_callable_config = True`           | `callable_editor` (Python rules)|
| FD JSON config   | `name == "consistency_countFDViolations"` (handled specially) | inline FD-rule editor |

`timeliness_heinrich` uses a dedicated `timeliness_editor` (selected by
metric name) because its config nests per-column settings.

#### GUI metadata class attributes

Declare these as class attributes on the `Metric` subclass. All are
optional and default to safe values; see existing metrics for examples.

| Attribute                        | Type           | Purpose                                                                                          |
|----------------------------------|----------------|--------------------------------------------------------------------------------------------------|
| `_gui_description`               | `str`          | Short summary of how the metric is calculated. Shown under the metric name in the GUI.           |
| `_gui_requires_reference`        | `bool`         | The metric needs a reference DataFrame (e.g. `correctness_heinrich`).                            |
| `_gui_config_required`           | `bool`         | The metric refuses to run without a config; the GUI blocks **Compute** until one is provided.    |
| `_gui_callable_config`           | `bool`         | The config carries Python callables (rules) and must be edited via the callable editor.          |
| `_gui_cell_granularity`          | `bool`         | The metric *can* emit per-cell results, so the GUI offers a row-limit cap.                       |
| `_gui_recommended_granularities` | `frozenset[DQGranularity]` | Granularities the metric produces meaningful results at. Used by the results page renderers. |

#### Native dependency declarations

Metrics that depend on a native library (e.g. FAHES) must register a check
in `_NATIVE_LIB_CHECKS` in `gui/core/metric_catalog.py`. The catalog will
mark the metric as unavailable when the library is missing, the GUI will
disable its checkbox with a warning, the per-dimension/global "Select all"
buttons will skip it, and `get_compute_blockers` will refuse to run it.

## Output: creating a DQResult

````python
class DQResult:
	def __init__(
		self,
		timestamp: pd.Timestamp,
		DQdimension: DQDimension,
		DQmetric: str,
		DQgranularity: DQGranularity,
		DQvalue: float,
		DQexplanation: Union[dict, None] = None,
		runtime: Union[float, None] = None,
		tableName: Union[str, None] = None,
		columnNames: Union[List[str], None] = None,
		rowIndex: Union[int, None] = None,
		experimentTag: Union[str, None] = None,
		dataset: Union[str, None] = None,
		configJson: Union[dict, None] = None,
	):
````

To create a new instance of DQResult, one needs to provide at least the following arguments:
- **timestamp: pd.Timestamp**: The time at which a result was assessed.
- **DQdimension: DQDimension**: Data quality dimension assessed (e.g. `DQDimension.COMPLETENESS`, `DQDimension.ACCURACY`).
- **DQmetric: str**: Name of the specific metric within the dimension.
- **DQgranularity: DQGranularity**: Granularity of the metric — one of `DQGranularity.CELL`, `DQGranularity.ROW`, `DQGranularity.COLUMN`, `DQGranularity.TABLE`.
- **DQvalue: float**: Numeric outcome of the assessment. This currently only supports quantitative assessments.

Furthermore, there are more optional arguments that might need to be set depending on the nature of different metrics. ```dataset``` and ```tableName``` are automatically set by the ```metis.dq_orchestrator.DQOrchestrator``` class which controls the data quality assessment and takes care of calling the individual metrics and storing the results.
- **DQexplanation: Optional[dict]**: Arbitrary additional information produced by the metric (no fixed schema required).
- **runtime: Optional[float]**: Time taken to compute the metric, in seconds.
- **columnNames: Optional[List[str]]**: Columns that this result pertains to. For a column-level metric this is typically a single-item list; for a table-level metric this may be `None` or an empty list.
- **rowIndex: Optional[int]**: Row index associated with the result. Use together with `columnNames` to represent a cell-level result, or for row-based metrics.
- **experimentTag: Optional[str]**: Tag to identify a specific run.
- **configJson: Optional[dict]**: Configuration used for the metric as a JSON object.

## Data Profiling

Metis includes a data profiling system that caches computed statistics and supports importing pre-computed profiles.

### Cached Profiling Functions

Use cached profiling functions from `metis.profiling` for automatic caching:

```python
from metis.profiling import null_count, distinct_count, data_type

# These are automatically cached when DataProfileManager is initialized
nulls = null_count(df["column"])
```

### Importing Pre-computed Profiles

You can import pre-computed data profiles (from external tools like HyFD, CFDFinder, etc.) via the data loader config:

```json
{
  "loader": "CSV",
  "name": "Adult",
  "file_name": "adult.csv",
  "data_profiles": {
	"fd": {
	  "source": "hyfd",
	  "file": "outputs/adult_hyfd.txt"
	},
	"null_count": {
	  "source": "manual",
	  "values": [
		{"column": "age", "value": 0},
		{"column": "workclass", "value": 1836}
	  ]
	}
  }
}
```

For complete documentation of all supported import formats, see [Data Profile Import Formats](docs/DATA_PROFILE_IMPORT_FORMATS.md).

### Cache Control Flags

Three flags can be passed to `DataProfileManager.initialize()`:

- **`ignore_cache`**: Never read from or write to the database. Pure passthrough on every call.
- **`overwrite_cache`**: Skip cache lookup; always recompute and overwrite the stored value. Note: every call recomputes, not just the first. There is no within-run caching.
- **`clear_cache`**: Delete all stored profiles at startup, then cache normally from there.

```python
DataProfileManager.initialize(engine, ignore_cache=True)    # passthrough, DB untouched
DataProfileManager.initialize(engine, overwrite_cache=True) # always recompute and overwrite
DataProfileManager.initialize(engine, clear_cache=True)     # wipe table at startup, then cache normally
```
