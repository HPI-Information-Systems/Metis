# Data Profile Import Formats

This document describes all supported formats for importing pre-computed data profiling results into Metis.

## Overview

Data profiles are defined in the data loader config (e.g., `adult.json`) under the `data_profiles` key. Each task type supports two import methods:

1. **Inline values** (`values`): Define results directly in the JSON config
2. **External file** (`file`): Reference a CSV or TXT file with the results

```json
{
  "loader": "CSV",
  "name": "Adult",
  "file_name": "adult.csv",
  "data_profiles": {
    "<task_name>": {
      "source": "<source_identifier>",
      "file": "path/to/file.csv",       // OR
      "values": [...]                    // inline values
    }
  }
}
```

---

## Cardinalities

### null_count, null_percentage, distinct_count, row_count, uniqueness

**JSON inline:**
```json
"null_count": {
  "source": "manual",
  "values": [
    {"column": "age", "value": 150},
    {"column": "income", "value": 42}
  ]
}
```

**CSV file:**
```csv
column,value
age,150
income,42
```

### value_length_min, value_length_max, value_length_mean, value_length_median

**JSON inline:**
```json
"value_length_max": {
  "source": "manual",
  "values": [
    {"column": "name", "value": 50},
    {"column": "email", "value": 100}
  ]
}
```

**CSV file:**
```csv
column,value
name,50
email,100
```

---

## Value Distribution

### constancy, interquartile_range

**JSON inline:**
```json
"constancy": {
  "source": "manual",
  "values": [
    {"column": "status", "value": 0.85}
  ]
}
```

**CSV file:**
```csv
column,value
status,0.85
```

### most_frequent_value

Values are auto-detected as int, float, bool, or string.

**JSON inline:**
```json
"most_frequent_value": {
  "source": "manual",
  "values": [
    {"column": "status", "value": "active"},
    {"column": "count", "value": 42}
  ]
}
```

**CSV file:**
```csv
column,value
status,active
count,42
```

### quartiles

**JSON inline:**
```json
"quartiles": {
  "source": "manual",
  "values": [
    {"column": "age", "Q1": 25.0, "Q2": 35.0, "Q3": 50.0},
    {"column": "income", "Q1": 30000, "Q2": 50000, "Q3": 80000}
  ]
}
```

**CSV file:**
```csv
column,Q1,Q2,Q3
age,25.0,35.0,50.0
income,30000,50000,80000
```

### equi_width_histogram, equi_depth_histogram

**JSON inline:**
```json
"equi_width_histogram": {
  "source": "manual",
  "values": [
    {
      "column": "age",
      "bins": [
        {"min": 0, "max": 30, "count": 1500},
        {"min": 30, "max": 60, "count": 2200},
        {"min": 60, "max": 90, "count": 800}
      ]
    }
  ]
}
```

**CSV file:**
```csv
column,bin_min,bin_max,count
age,0,30,1500
age,30,60,2200
age,60,90,800
income,0,50000,3000
income,50000,100000,2500
```

---

## Patterns and Data Types

### basic_type, data_type, data_class, domain

**JSON inline:**
```json
"basic_type": {
  "source": "manual",
  "values": [
    {"column": "age", "value": "numeric"},
    {"column": "name", "value": "alphabetic"}
  ]
}
```

**CSV file:**
```csv
column,value
age,numeric
name,alphabetic
```

Valid values for `basic_type`: `numeric`, `alphabetic`, `alphanumeric`, `date`, `time`, `mixed`, `empty`

Valid values for `data_type`: `boolean`, `smallint`, `int`, `bigint`, `numeric`, `double`, `date`, `time`, `timestamp`, `varchar`, `text`

Valid values for `data_class`: `code`, `indicator`, `text`, `date/time`, `quantity`, `identifier`

Valid values for `domain`: `email`, `url`, `ssn`, `date_iso`, `time`, `ip_address`, `zip_code`, `credit_card`, `phone`, `currency`, `first_name`, `last_name`, `full_name`, `city`, `state`, `country`, `address`, `postal_code`, `unknown`

### size, decimals

**JSON inline:**
```json
"size": {
  "source": "manual",
  "values": [
    {"column": "price", "value": 10}
  ]
}
```

**CSV file:**
```csv
column,value
price,10
```

### patterns

**JSON inline:**
```json
"patterns": {
  "source": "manual",
  "values": [
    {
      "column": "phone",
      "patterns": [
        {"pattern": "999-999-9999", "count": 500, "frequency": 0.8},
        {"pattern": "(999) 999-9999", "count": 100, "frequency": 0.16}
      ]
    }
  ]
}
```

**CSV file:**
```csv
column,pattern,count,frequency
phone,999-999-9999,500,0.8
phone,(999) 999-9999,100,0.16
```

Pattern codes: `A`=uppercase, `a`=lowercase, `9`=digit, `#`=special char, `?`=other letter, ` `=space

---

## Summaries and Sketches

### jaccard_similarity

**JSON inline:**
```json
"jaccard_similarity": {
  "source": "manual",
  "values": [
    {"column1": "name", "column2": "alias", "value": 0.85}
  ]
}
```

**CSV file:**
```csv
column1,column2,value
name,alias,0.85
```

### jaccard_similarity_ngrams

**JSON inline:**
```json
"jaccard_similarity_ngrams": {
  "source": "manual",
  "values": [
    {"column1": "name", "column2": "alias", "n": 2, "value": 0.78}
  ]
}
```

**CSV file:**
```csv
column1,column2,n,value
name,alias,2,0.78
```

### minhash_signature

Not importable (returns MinHash objects).

---

## Dependencies

### Functional Dependencies (fd)

#### JSON inline

```json
"fd": {
  "source": "manual",
  "values": [
    {"lhs": ["zip"], "rhs": "city"},
    {"lhs": ["id"], "rhs": "name"},
    {"lhs": ["city", "street"], "rhs": "zip"}
  ]
}
```

#### External file (HyFD / AIDFD format)

```json
"fd": {
  "source": "hyfd",
  "file": "outputs/adult_hyfd.txt"
}
```

**HyFD/AIDFD output format:**
```
[table.col1]->table.col2
[table.col1, table.col2]->table.col3
```

Each FD on the same line, space-separated. Table prefix is stripped automatically.

#### External file (CFDFinder format)

```json
"fd": {
  "source": "cfdfinder",
  "file": "outputs/adult_cfd.txt"
}
```

**CFDFinder output format:**
```
[table.col1]->table.col2#(pattern1);(pattern2)
```

The pattern tableau is stored as additional metadata.

### Unique Column Combinations (ucc)

**JSON inline:**
```json
"ucc": {
  "source": "manual",
  "values": [
    {"columns": ["id"]},
    {"columns": ["name", "birthdate"]}
  ]
}
```

**CSV file:**
```csv
columns
id
"name,birthdate"
```

### Inclusion Dependencies (ind)

**JSON inline:**
```json
"ind": {
  "source": "manual",
  "values": [
    {
      "dependent": ["customer_id"],
      "referenced": ["id"],
      "referenced_table": "customers"
    }
  ]
}
```

**CSV file:**
```csv
dependent,referenced,referenced_table
customer_id,id,customers
"order_id,product_id","id,id","orders,products"
```

---

## Source Identifiers

The `source` field tracks where the data came from:

| Source | Description |
|--------|-------------|
| `manual` | Manually entered values |
| `hyfd` | HyFD algorithm output |
| `aidfd` | AIDFD algorithm output |
| `cfdfinder` | CFDFinder algorithm output |
| `computed` | Computed by Metis (automatic) |
| `imported:<tool>` | Custom import source |

---

## Example: Complete Config

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
    },
    "equi_width_histogram": {
      "source": "manual",
      "file": "outputs/adult_histograms.csv"
    },
    "basic_type": {
      "source": "manual",
      "values": [
        {"column": "age", "value": "numeric"},
        {"column": "education", "value": "alphabetic"}
      ]
    }
  }
}
```
