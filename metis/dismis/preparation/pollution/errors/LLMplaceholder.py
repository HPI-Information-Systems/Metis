from typing import List
import numpy as np
import pandas as pd
import random
import json
import re

from .error import DMV

class LLMPlaceholderDMV2(DMV):
    prompt = """Your task is to act as a data quality analyst to find Disguised Missing Values (DMVs).

Your process is in two steps:
1.  **Analysis:** First, critically examine the 5 example values provided from the column. Reason about which of them seem like valid data points and which might be placeholders. Briefly explain your reasoning.
2.  **Generation:** Based on your analysis of what the **valid** values look like, generate a list of 20 potential DMVs. These DMVs must be semantically and syntactically distinct from the values you identified as valid.

Generate the DMVs in two categories:
-   **Generic:** Common placeholders (e.g., 'N/A').
-   **Context-Specific:** Placeholders derived from the table and column names (e.g., 'Season cancelled'). This is the most important category. This should include identified potential placeholders fromt the input data.

IMPORTANT: Follow the JSON format exactly as shown in the example below. Use the same keys.

# Example 1:
Table name: "students"
Column name: "test_score"
## Example values to analyze:
- 85
- 92
- Not Graded
- 78
- 65

# Your Response:
## 1. Analysis:
- 85, 92, 78, 65: These look like **Likely Valid** values. They are typical integer scores for a test.
- Not Graded: This is a **Potential Placeholder**. It's a text string explaining why a numerical score is missing.
### Summary:
{{
    "Valid": [85, 92, 78, 65],
    "Placeholders": ["Not Graded"]
}}

## 2. Potential placeholder values:
{{
    "Generic": ["N/A", "Unknown", "U", "Missing", "?", "-", "None", "Null", "To Be Determined", "TBD", "Not Available"],
    "Context-Specific": ["Not Graded", "Absent", "Incomplete", "Withdrew from course", "Pending grade", "Exempt", "There was no test this semester", "Test not taken", "Score not recorded", "Grade pending", "-1"]
}}

---

# Input:
Table name: "{table_name}"
Column name: "{column_name}"
## Example values to analyze:
"""

    last_line = "# Your Response:\n"

    def __init__(self, LLM, table_name: str, repeating: bool = True, ):
        """
        Initialize the PlaceholderDMV with optional placeholder values.
        Args:
            placeholder_values (Union[None, List[str], dict]):
                A list of placeholder values to choose from. If None, defaults to ["N/A", "Unknown", "Placeholder"]. If a dictionary is provided, it should map column names to lists of placeholder values.
            repeating (bool):
                If True, allows the same placeholder value to be used multiple times in the same column. If False, ensures that each placeholder value is unique within a column.
        """

        self.repeating = repeating
        self.table_name = table_name
        self.LLM = LLM

    def get_column_placeholders(self, column_names: List[str], example_values: dict) -> tuple[dict, dict, dict]:
        """
        Get column placeholders and judged example values.

        Returns:
            tuple: (col_placeholders, valid_values, invalid_values)
                - col_placeholders: dict mapping column names to lists of generated placeholder values
                - valid_values: dict mapping column names to lists of example values judged as valid
                - invalid_values: dict mapping column names to lists of example values judged as invalid/placeholders
        """
        col_placeholders = {col: [] for col in column_names}
        valid_values = {col: [] for col in column_names}
        invalid_values = {col: [] for col in column_names}

        while any(len(placeholders) < 10 for placeholders in col_placeholders.values()):
            remaining_cols = [col for col in column_names if len(col_placeholders[col]) < 10]
            all_messages = []
            for col in remaining_cols:
                prompt = self.prompt.format(column_name=col, table_name=self.table_name)
                for value in example_values[col]:
                    prompt += f"- {value}\n"

                prompt += self.last_line
                messages = [
                    {"role": "system", "content": "You are a data engineer working on quality control of tabular data."},
                    {"role": "user", "content": prompt},
                ]
                all_messages.append(messages)


            responses = self.LLM.generate(all_messages)
            # for messages, response in zip(all_messages, responses):
            #     print("---- Prompt ----")
            #     print(messages[-1]['content'])
            #     print("---- Response ----")
            #     print(response)
            #     print("---- End Response ----")
            # Extract the list of placeholder values from the response
            for col, response in zip(remaining_cols, responses):
                # Extract values here
                # Try to find JSON blocks in the response
                # Look for both the analysis summary (first JSON) and placeholder values (second JSON)

                try:
                    # Find all JSON-like objects in the response
                    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                    json_matches = re.findall(json_pattern, response, re.DOTALL)

                    if len(json_matches) >= 2:
                        # First JSON: Analysis summary with Valid and Placeholders
                        analysis_json = json.loads(json_matches[0])
                        valid = analysis_json.get("Valid", [])
                        invalid = analysis_json.get("Placeholders", [])

                        # Store the judged values
                        for val in valid:
                            if val not in valid_values[col]:
                                valid_values[col].append(val)
                        for val in invalid:
                            if val not in invalid_values[col]:
                                invalid_values[col].append(val)

                        # Second JSON: Placeholder values with Generic and Context-Specific
                        placeholder_json = json.loads(json_matches[1])

                        # Extract Generic and Context-Specific placeholder values
                        generic = placeholder_json.get("Generic", [])
                        context_specific = placeholder_json.get("Context-Specific", [])

                        if not isinstance(generic, list):
                            print("Warning: 'Generic' placeholders is not a list.", generic)
                            generic = []

                        if not isinstance(context_specific, list):
                            print("Warning: 'Context-Specific' placeholders is not a list.", context_specific)
                            context_specific = []

                        # Combine both categories
                        new_placeholders = generic + context_specific

                        for placeholder in new_placeholders:
                            if placeholder and placeholder not in col_placeholders[col]:# and placeholder not in valid_values[col]:
                                col_placeholders[col].append(placeholder)

                except (json.JSONDecodeError, IndexError, KeyError) as e:
                    # If JSON parsing fails, try to extract manually
                    # Look for lines that might contain placeholder values after "Generic" or "Context-Specific"
                    if '"Generic"' in response or '"Context-Specific"' in response:
                        # Try to extract array values from the JSON
                        array_pattern = r'\[(.*?)\]'
                        arrays = re.findall(array_pattern, response, re.DOTALL)
                        if len(arrays) == 4:
                            for array_str in arrays[2:]:
                                # Extract quoted strings from the array
                                values = re.findall(r'"([^"]*)"', array_str)
                                for value in values:
                                    if value and value not in col_placeholders[col]:
                                        col_placeholders[col].append(value)

        return col_placeholders, valid_values, invalid_values

    def __call__(self, dataset: pd.DataFrame, positions: np.ndarray) -> pd.DataFrame:
        """
        Introduce placeholder values into the dataset at the specified positions.

        Args:
            dataset (pd.DataFrame): The dataset to introduce placeholders into.
            positions (np.ndarray): Array of positions where placeholders should be introduced.

        Returns:
            pd.DataFrame: The dataset with placeholders introduced.
        """

        columns = dataset.columns.to_list()
        example_values = {col: list(set(random.sample(dataset[col][:10000].dropna().astype(str).tolist(), 20)))[:5]
                  for col in columns}
        column_placeholder_values, valid_values, invalid_values = self.get_column_placeholders(columns, example_values)

        # Store the judged values as instance attributes for later access
        self.valid_values = valid_values
        self.invalid_values = invalid_values

        for column_idx in range(len(columns)):
            col_positions = positions[positions[:, 1] == column_idx]
            if len(col_positions) > 0:
                placeholder_values = column_placeholder_values[columns[column_idx]]

                if self.repeating:
                    placeholder = random.choice(placeholder_values)
                    rows = col_positions[:, 0]
                    dataset.iloc[rows, column_idx] = placeholder

                else:
                    # Shuffle the positions
                    np.random.shuffle(col_positions)
                    # Split into n roughly equal chunks
                    chunks = np.array_split(col_positions, len(placeholder_values))
                    for chunk, placeholder in zip(chunks, placeholder_values):
                        rows = chunk[:, 0]
                        dataset.iloc[rows, column_idx] = placeholder

        return dataset

class LLMNonsenseDMV2(LLMPlaceholderDMV2):
    prompt = """Your task is to act as a data quality analyst to find Nonsense Placeholder Values (NPVs).

Your process is in two steps:
1.  **Analysis:** First, critically examine the 5 example values provided from the column. Reason about which of them seem like valid data points and which might be placeholders. Briefly explain your reasoning.
2.  **Generation:** Based on your analysis of what the **valid** values look like, generate a list of 20 potential NPVs. These NPVs must be semantically and syntactically distinct from the values you identified as valid.

Generate the NPVs in two categories:
-   **Generic:** Common nonsense values (e.g., 'asdfg').
-   **Context-Specific:** Nonsense values derived from the table and column names (e.g., 'some test result'). This is the most important category.

IMPORTANT: Follow the JSON format exactly as shown in the example below. Use the same keys.

# Example 1:
Table name: "students"
Column name: "test_score"
## Example values to analyze:
- 85
- 92
- sdfgh
- 78
- 65

# Your Response:
## 1. Analysis:
- 85, 92, 78, 65: These look like **Likely Valid** values. They are typical integer scores for a test.
- sdfgh: This is a **Potential Nonsense**. Its a string consisting of random letters that does not convey any meaningful information about a test score.
### Summary:
{{
    "Valid": [85, 92, 78, 65],
    "Placeholders": ["sdfgh"]
}}

## 2. Potential nonsense values:
{{
    "Generic": ["asdfg", "?????", "aaaaaa", "hi", "SCORE", "lol"],
    "Context-Specific": ["testtesttest", "scooore", "some test result", "my score is", "the score is", "score:", "123abc", "score here", "the result is", "score unknown", "sdfgh"]
}}

---

# Input:
Table name: "{table_name}"
Column name: "{column_name}"
## Example values to analyze:
"""

class LLMCommentDMV2(LLMPlaceholderDMV2):
    prompt = """Your task is to act as a data quality analyst to find comment values in the dataset.

Your process is in two steps:
1.  **Analysis:** First, critically examine the 5 example values provided from the column. Reason about which of them seem like valid data points and which might be comments. Briefly explain your reasoning.
2.  **Generation:** Based on your analysis of what the **valid** values look like, generate a list of 20 potential comments. These comments must be semantically and syntactically distinct from the values you identified as valid.

Generate the comments in two categories:
-   **Generic:** Common comments (e.g., 'I will update this later').
-   **Context-Specific:** Comments derived from the table and column names (e.g., 'I forgot to ask for the address'). This is the most important category.

IMPORTANT: Follow the JSON format exactly as shown in the example below. Use the same keys.

# Example 1:
Table name: "students"
Column name: "test_score"
## Example values to analyze:
- 85
- 92
- student dropped out
- 78
- 65

# Your Response:
## 1. Analysis:
- 85, 92, 78, 65: These look like **Likely Valid** values. They are typical integer scores for a test.
- student dropped out: This is a **Potential Comment**. It's a text explaining why a numerical score is missing.
### Summary:
{{
    "Valid": [85, 92, 78, 65],
    "Placeholders": ["student dropped out"]
}}

## 2. Potential comment values:
{{
    "Generic": ["have to look this up", "I will update this later", "Needs confirmation", "Please verify", "Will fill in later", "I dont know"],
    "Context-Specific": ["student absent", "test not taken", "score pending", "I forgot to ask for the score", "need to check with student", "not sure about the score", "score to be determined", "awaiting student response", "test cancelled", "student dropped out"]
}}

---

# Input:
Table name: "{table_name}"
Column name: "{column_name}"
## Example values to analyze:
"""

class LLMUnsureDMV2(LLMPlaceholderDMV2):
    prompt = """Your task is to act as a data quality analyst to find values in the dataset marked as unsure.

Your process is in two steps:
1.  **Analysis:** First, critically examine the 5 example values provided from the column. Reason about which of them seem like valid data points and which might be marked as unsure. Briefly explain your reasoning.
2.  **Generation:** Based on your analysis of what the **valid** values look like, generate a list of 20 potential unsure values. These comments must be semantically and syntactically distinct from the values you identified as valid.

Generate the comments in two categories:
-   **Generic:** Common comments (e.g., '??').
-   **Context-Specific:** Comments derived from the table and column names (e.g., '1234 Fake St (I guess)'). This is the most important category.

IMPORTANT: Follow the JSON format exactly as shown in the example below. Use the same keys.

# Example 1:
Table name: "students"
Column name: "test_score"
## Example values to analyze:
- 85
- 92
- around 80
- 78
- 65

# Your Response:
## 1. Analysis:
- 85, 92, 78, 65: These look like **Likely Valid** values. They are typical integer scores for a test.
- around 80: This is a **Potential Unsure Value**. 'around' makes this score less precise.
### Summary:
{{
    "Valid": [85, 92, 78, 65],
    "Placeholders": ["around 80"]
}}

## 2. Potential unsure values:
{{
    "Generic": ["??", "approximately", "around", "not sure", "I think", "maybe", "(?)", "I guess", "close to", "roughly"],
    "Context-Specific": ["85?", "92 (not sure)", "around 78", "I think it is 65", "maybe 80", "78 (?)", "close to 90", "approximately 70", "I guess 75", "roughly 88"]
}}

---

# Input:
Table name: "{table_name}"
Column name: "{column_name}"
## Example values to analyze:
"""

class LLMValidDMV2(LLMPlaceholderDMV2):
    prompt = """Your task is to act as a data quality analyst to find valid values in a dataset.

Your process is in two steps:
1.  **Analysis:** First, critically examine the 5 example values provided from the column. Reason about which of them seem like valid data points and which might be placeholders or default values. Briefly explain your reasoning.
2.  **Generation:** Based on your analysis of what the **valid** values look like, generate a list of 20 potential valid values occurring in this column. These values must be semantically and syntactically distinct from the values you identified as invalid.

Generate the values in two categories:
-   **Generic:** Common valid values in such a column (e.g., 'yes').
-   **Context-Specific:** Comments derived from the table and column names (e.g., 'positive test result'). This is the most important category.

IMPORTANT: Follow the JSON format exactly as shown in the example below. Use the same keys.

# Example 1:
Table name: "students"
Column name: "exam_attendance"
## Example values to analyze:
- yes
- no
- unsure
- attended
- none

# Your Response:
## 1. Analysis:
- yes, no, attended: These look like **Likely Valid** values. They are typical responses for an attendance column.
- unsure, none: These look like **Likely Invalid** values. It indicates a lack of certainty about attendance.
### Summary:
{{
    "Valid": ["yes", "no", "attended"],
    "Placeholders": ["unsure", "none"]
}}

## 2. Potential valid values:
{{
    "Generic": ["yes", "no"],
    "Context-Specific": ["did not attend", "attended", "present", "absent", "confirmed", "not confirmed", "attending", "not attending"]
}}

---

# Input:
Table name: "{table_name}"
Column name: "{column_name}"
## Example values to analyze:
"""
