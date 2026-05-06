"""Renderer for rule-based violation metrics.

Triggered when explanation keys contain ``:`` (e.g. ``"{determinant}:{dependent}"``
for functional-dependency violations). Detection is purely data-driven — no
metric name is examined anywhere in this module.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st


def extract_rules(results: list[dict]) -> dict[str, list]:
    """
    Extract per-rule violation data from a list of result dicts.

    Keys containing ``:`` are treated as rule descriptors of the form
    ``"{det}:{dep}"``.

    :param results: A list of result dicts as produced by ``get_table_results``.
    :return: Mapping ``rule_key → list of violating determinant values``.
    """
    rules: dict[str, list] = {}
    for r in results:
        for k, v in (r.get("DQexplanation") or {}).items():
            if ":" in k and k not in rules:
                rules[k] = v if isinstance(v, list) else []
    return rules


def render_violations_table(rules: dict[str, list], cache_key: str = "") -> None:
    """
    Render a selectable table of absolute violation counts per rule.

    Selecting a row reveals the actual determinant values that violate that rule.

    :param rules: Mapping ``rule_key → violations_list``.
    :param cache_key: Stable cache key for widget state.
    :return: None.
    """
    if not rules:
        st.warning("No results to display.")
        return

    rows = []
    for rule_key, violations in rules.items():
        det, dep = rule_key.split(":", 1)
        rows.append({
            "Rule": f"{det} → {dep}",
            "Violations": len(violations),
            "_violations": violations,
        })

    full_df = (
        pd.DataFrame(rows)
        .sort_values("Violations", ascending=False)
        .reset_index(drop=True)
    )

    event = st.dataframe(
        full_df[["Rule", "Violations"]],
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key=f"fd_violations_table_{cache_key}",
    )

    selected = event.selection.rows if event.selection else []
    if not selected:
        st.caption("Select one or more rows to inspect the violating values.")
        return

    for idx in selected:
        row = full_df.iloc[idx]
        rule_label = row["Rule"]
        violations = row["_violations"]

        if not violations:
            st.caption(f"No violating values for `{rule_label}`.")
            continue

        st.markdown(f"**Violating values for `{rule_label}`** ({len(violations)})")
        st.dataframe(
            pd.DataFrame({"Determinant value": violations}),
            width="stretch",
            hide_index=True,
        )
