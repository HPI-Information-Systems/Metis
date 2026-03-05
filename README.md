# Metis

Metis is a framework to automatically assess the quality of tabular data across multiple data quality dimensions. The Metis DQ framework (this GitHub repo) is part of the Metis project: [www.metisdq.org](https://www.metisdq.org)

![Overview](images/overview.png)

## Start the demo and run Metis:
```
python -m demo.getting_started
```

## How to implement new metrics

To extend the Metis framework and add new data quality metrics, please check our interface for easy integration. 
````python
def assess(self, 
            data: pd.DataFrame, 
            reference: Union[pd.DataFrame, None] = None, 
            metric_config: Union[str, None] = None) -> List[DQResult]:
````
Each metric should be a subclass of ```metis.metric.metric.Metric``` and implement the assess method. This method takes three arguments:
- **data: pandas.Dataframe**: The DataFrame that should be assessed by this metric. This is the primary dataset under inspection.
- **reference: Optional[pd.DataFrame]**: An optional, cleaned reference DataFrame that can act as a gold-standard / ground-truth version of the dataset. Metrics that need a clean version of the data (e.g., correctness against a known-good source) should accept and use this DataFrame. If not needed by a metric, `None` is allowed.
- **metric_config: Optional[str]**: Optional path or JSON string containing metric-specific configuration. Use this to keep the method signature compact; all metric-specific parameters (thresholds, aggregation options, etc.) can be stored here.

The metric should return a list of ```metis.utils.result.DQResult```. This can be only one object if one value is computed on a table level or mutliple DQResults if for example one result per column is computed.

**Note:** Each metric has to be imported in the *__init__.py* file inside the folder *metric/* so it is recognized by the Metric registry.

### Metric naming convention

Metrics are organized by dimension (e.g., `completeness`, `minimality`), where one folder exists for each. 
New metrics should follow the naming format: `{DimensionName}_{Technique}`

- **DimensionName**: The quality dimension being measured (e.g., `Completeness`, `Minimality`)
- **Technique**: The calculation or method used (e.g., `NullRatio`, `DuplicateCount`)
- **Granularity**: The level of analysis (e.g., `cell`, `row`, `column`, `table`) should be passed as a parameter through the metric config file if the metric can be applied at different granularity levels.

Examples: `completeness_NullRatio`, `minimality_DuplicateCount`

## Output: creating a DQResult

````python
class DQResult:
    def __init__(
        self,
        timestamp: pd.Timestamp,
        DQvalue: float,
        DQdimension: str,
        DQmetric: str,
        columnNames: Union[List[str], None] = None,
        rowIndex: Union[int, None] = None,
        DQannotations: Union[dict, None] = None,
        dataset: Union[str, None] = None,
        tableName: Union[str, None] = None,
    ):
````

To create a new instance of DQResult, one needs to provide at least the following arguments:
- **timestamp: pd.Timestamp**: The time at which a result was assessed.
- **DQvalue: float**: The result of the assessment. This currently only supports quantitative assessments.
- **DQdimension: str**: The name of the data quality dimension that was assessed e.g. completeness, accuracy, etc.
- **DQmetric: str**: The name of the specific metric inside the given dimension that was assessed.

Furthermore, there are more optional arguments that might need to be set depending on the nature of different metrics. ```dataset``` and ```tableName``` are automatically set by the ```metis.dq_orchestrator.DQOrchestrator``` class which controles the data quality assessment and takes care of calling the individual metrics and storing the results.
- **columnNames: Optional[List[str]]**: List of column names associated with the assessed result. For example for column level completeness, this would be a list with a single column name, for table level completeness this would be empty since the result is valid for the whole table.
- **rowIndex: Optional[int]**: Index of the row this result is associated with. This can either be used together with columnNames to assess data quality on a cell level or for row based metrics.
- **DQannotations: Optional[dict]**: To allow metrics to save additional information or annotations, this dictionary can store all additional information that might need to be saved. This currently does not need for follow a predefined structure.

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

