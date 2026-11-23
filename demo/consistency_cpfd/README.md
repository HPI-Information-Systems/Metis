# consistency_cpfd demo

Runnable example for the `consistency_cpfd` metric (Seeger et al., VLDB
QDB'26). The metric is fully automated: it consumes a set of partial
functional dependencies (pFDs) with their `partial` threshold (rho) and
`gpdep` genuineness weight, and aggregates them to a single consistency score

```
cpfd(F) = sum_{X->A in F} omega(X,A) * r(X,A) / sum_{X->A in F} omega(X,A)
```

where `r(X,A) = 1` iff `partial = 1`.

## Files

- [`hospital_pfds.txt`](hospital_pfds.txt) — Partial HyFD output on the
  cleaned `hospital` dataset (20 pFDs, the example used in the paper).
  One line per pFD in the format

  ```
  [<table>.<file_ext>.<lhs_col>, ...]-><table>.<file_ext>.<rhs_col>#<partial>#<gpdep>
  ```

  Empty LHS (`[]->...`) marks a constant-column claim and is kept with
  `lhs = []`.

## Running the metric

Add the pFDs to the data-loader config for the dataset:

```json
{
  "loader": "CSV",
  "name": "Hospital",
  "file_name": "data/hospital.csv",
  "data_profiles": {
    "pfd": {
      "source": "cpfd",
      "file": "demo/consistency_cpfd/hospital_pfds.txt"
    }
  }
}
```

Then run the metric via the orchestrator:

```python
from metis.dq_orchestrator import DQOrchestrator

orchestrator = DQOrchestrator()
orchestrator.load(["path/to/hospital_loader.json"])
results = orchestrator.assess(metrics=["consistency_cpfd"])
for r in results:
    print(r.as_json())
```

Note: `consistency_cpfd.assess()` does not read the input DataFrame —
consistency is fully determined by the imported pFDs. The metric does
require that `DataProfileManager` is initialized and that the dataset/table
context matches the one under which the pFDs were imported.

## Expected output

A single `DQResult` at `DQGranularity.TABLE`:

- `DQvalue = 1.0` (all 20 pFDs in the example are exact, `partial = 1.0`,
  so `r = 1` for every pFD and the score collapses to `sum(ω) / sum(ω) = 1`).
- `DQexplanation["num_pfds"] = 20`.
- `DQexplanation["pfds"]` lists each pFD with its `lhs`, `rhs`, `partial`,
  `gpdep`, and `r`.

Editing one of the lines to a non-exact `partial` (e.g. `#0.9#…`) lowers
the score in proportion to that rule's `gpdep` weight — a quick way to
sanity-check the metric against the formula by hand.

## Producing your own pFDs

Run the Partial HyFD extension from
[SeegerM/cpfd-reproducibility](https://github.com/SeegerM/cpfd-reproducibility)
on the dataset and feed its output file directly to the `pfd` importer.
The text format is documented above and in
[`docs/DATA_PROFILE_IMPORT_FORMATS.md`](../../docs/DATA_PROFILE_IMPORT_FORMATS.md#partial-functional-dependencies-pfd).
