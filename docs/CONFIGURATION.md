# Configuration reference

Metis is configured through small JSON files of three kinds:

- **Data loader configs** describe how a dataset is read into a DataFrame.
- **Writer configs** describe where assessment results are written.
- **Metric configs** carry metric-specific parameters and are documented
  with each metric (see
  [How to implement new metrics](../README.md#how-to-implement-new-metrics)).

This file is the field reference for the first two, plus a short note on
how metric configs are passed.

## Data loader configs

A data loader config is passed to `DQOrchestrator.load()`:

```python
orchestrator.load(data_loader_configs=["data/restaurants.json"])
```

Example with all common fields:

```json
{
    "loader": "CSV",
    "name": "AdultWithReference",
    "file_name": "adult.csv",
    "reference_file_name": "adult_gold_sample.csv",
    "delimiter": ",",
    "encoding": "utf-8",
    "header": 0,
    "nrows": 100,
    "usecols": ["age", "workclass", "education"],
    "parse_dates": false,
    "decimals": ".",
    "thousands": null
}
```

### Fields

| Field                 | Required | Default   | Meaning                                                              |
|-----------------------|----------|-----------|----------------------------------------------------------------------|
| `name`                | yes      |           | Dataset identifier. Used as the table name on results.               |
| `file_name`           | yes      |           | CSV path, resolved relative to the data root (`data/`).              |
| `loader`              | yes      |           | Loader type. `CSV` is currently the only supported value.            |
| `reference_file_name` | no       | `null`    | Optional reference CSV, also relative to `data/` (see below).        |
| `delimiter`           | no       | `","`     | Column separator.                                                    |
| `encoding`            | no       | `"utf-8"` | File encoding.                                                       |
| `header`              | no       | `0`       | Row number to use as the header.                                     |
| `nrows`               | no       | `null`    | Read only the first n rows.                                          |
| `usecols`             | no       | `null`    | List of column names to load.                                        |
| `parse_dates`         | no       | `false`   | Forwarded to `pandas.read_csv`.                                      |
| `decimals`            | no       | `"."`     | Decimal separator. Note the key is `decimals`, not pandas' `decimal`.|
| `thousands`           | no       | `null`    | Thousands separator.                                                 |
| `data_profiles`       | no       | `null`    | Pre-computed profile imports (see below).                            |

The data root is defined in `metis/globals.py` (`data_root = "data"`), so
`"file_name": "adult.csv"` resolves to `data/adult.csv` regardless of the
config file's own location.

### Reference datasets

When `reference_file_name` is set, the orchestrator loads that file with
the same parsing options as the primary file and passes it as the
`reference` argument to every metric run on this dataset. Reference-based
metrics such as `correctness_heinrich` and `accuracy_semanticReference`
require it and do nothing useful without one. See
`demo/configs/adult_with_reference.json` for a working example.

### Pre-computed data profiles

The optional `data_profiles` key imports profiling statistics computed by
external tools (HyFD, CFDFinder, and others) or supplied manually, so that
metrics can use them instead of recomputing:

```json
{
    "data_profiles": {
        "fd": {"source": "hyfd", "file": "outputs/adult_hyfd.txt"},
        "null_count": {
            "source": "manual",
            "values": [{"column": "age", "value": 0}]
        }
    }
}
```

Imports require a database-backed writer (SQLite or PostgreSQL), because
profiles are stored in the same database as the profiling cache. The full
format reference for every supported task and source lives in
[DATA_PROFILE_IMPORT_FORMATS.md](DATA_PROFILE_IMPORT_FORMATS.md).

## Writer configs

A writer config is passed to the orchestrator constructor:

```python
orchestrator = DQOrchestrator(writer_config_path="configs/writer/sqlite.json")
```

The `writer_name` field selects the writer. Without a
`writer_config_path`, results are printed to the console. Example configs
live in `configs/writer/`.

### Console (default)

No config needed. Each result is printed as JSON to stdout.

### CSV

```json
{
    "writer_name": "csv",
    "path": "output/dq_results.csv"
}
```

`path` must end with `.csv`. An existing file is overwritten, not
appended to. Parent directories are created automatically.

### SQLite

```json
{
    "writer_name": "sqlite",
    "table_name": "dqresults",
    "db_name": "dq_repository/dq_repository.db"
}
```

`db_name` is the path of the database file. `table_name` is optional and
defaults to `dq_results`.

### PostgreSQL

```json
{
    "writer_name": "postgres",
    "table_name": "dqresults",
    "db_user": "postgres",
    "db_pass": "postgres",
    "db_name": "metis_db",
    "db_host": "localhost",
    "db_port": 5432
}
```

All of `db_user`, `db_pass`, `db_name`, `db_host` and `db_port` are
required. A matching local database can be started with Docker:

```
docker compose -f docker_compose.yaml up -d
```

The compose file provisions PostgreSQL with the credentials shown above.

### Behavior notes

- Choosing `sqlite` or `postgres` initializes the central `Database`
  singleton, which also enables the data profiling cache (see
  [Data Profiling](../README.md#data-profiling)). Only one database can be
  initialized per process, so a second orchestrator in the same script
  must use the console or CSV writer.
- If writing results fails, the orchestrator falls back to dumping them
  into `dq_results_fallback.csv` in the working directory, so a failed
  database connection does not lose a finished assessment.

## Passing metric configs

`DQOrchestrator.assess()` takes `metrics` and `metric_configs` as parallel
lists. Each entry of `metric_configs` may be:

- `None` or `""` for the metric's defaults
- a path to a `.json` file (must end with `.json`)
- a JSON string, for example `'{"use_nltk": true}'`
- a pre-instantiated config object, for example
  `completeness_nullRatio_config(aggregate_all=True)`

One flag is understood across all metrics: setting `"measure_runtime":
true` in a metric config makes the orchestrator time the metric and store
the elapsed seconds in the `runtime` field of each result.
