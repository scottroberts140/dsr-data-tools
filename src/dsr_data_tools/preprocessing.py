from __future__ import annotations

from typing import Any

import pandas as pd


def apply_preprocessing(
    df: pd.DataFrame, preprocessing_config: dict[str, dict[str, Any]] | None
) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply column-level preprocessing operations to a DataFrame.

    Supported operations
    --------------------
    group_by_frequency
        Keep the top-N most frequent categories and collapse all others
        into an ``other_label`` bucket.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    preprocessing_config : dict[str, dict[str, Any]] | None
        Mapping of column name to preprocessing directives.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        The transformed DataFrame and status messages describing operations.
    """
    msgs: list[str] = []
    if not preprocessing_config:
        return df.copy(), msgs

    out_df = df.copy()

    for column_name, column_config in preprocessing_config.items():
        if column_name not in out_df.columns:
            msgs.append(f"⚠️ Skipped preprocessing for missing column '{column_name}'.")
            continue

        if not isinstance(column_config, dict):
            msgs.append(
                f"⚠️ Skipped preprocessing for '{column_name}': expected a dict config."
            )
            continue

        group_config = column_config.get("group_by_frequency")
        if group_config is not None:
            out_df, group_msgs = _apply_group_by_frequency(
                out_df, column_name, group_config
            )
            msgs.extend(group_msgs)

    return out_df, msgs


def _apply_group_by_frequency(
    df: pd.DataFrame, column_name: str, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    """Apply top-N categorical grouping for a single column."""
    msgs: list[str] = []

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid 'group_by_frequency' config for '{column_name}': expected a mapping."
        )

    top_n = config.get("top_n", 10)
    other_label = config.get("other_label", "Other")

    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError(
            f"Invalid top_n for '{column_name}': expected int >= 1, got {top_n!r}."
        )

    value_counts = df[column_name].value_counts(dropna=False)
    unique_count = len(value_counts)

    if unique_count <= top_n:
        msgs.append(
            f"ℹ️ {column_name}: no grouping needed ({unique_count} unique values <= top_n={top_n})."
        )
        return df, msgs

    top_values = value_counts.head(top_n).index
    grouped_df = df.copy()
    grouped_df[column_name] = grouped_df[column_name].where(
        grouped_df[column_name].isin(top_values), other_label
    )

    grouped_count = unique_count - top_n
    msgs.append(
        f"🧩 {column_name}: grouped {grouped_count} categories into '{other_label}' (top_n={top_n})."
    )
    return grouped_df, msgs
