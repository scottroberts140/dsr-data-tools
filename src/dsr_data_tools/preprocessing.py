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
    min_frequency
        Collapse categories whose frequency (count or proportion) falls
        below a threshold into an ``other_label`` bucket.
    bucketing
        Discretize a numeric column into explicit, user-defined bins using
        provided edges and optional labels.

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

        min_freq_config = column_config.get("min_frequency")
        if min_freq_config is not None:
            out_df, min_freq_msgs = _apply_min_frequency(
                out_df, column_name, min_freq_config
            )
            msgs.extend(min_freq_msgs)

        bucketing_config = column_config.get("bucketing")
        if bucketing_config is not None:
            out_df, bucketing_msgs = _apply_bucketing(
                out_df, column_name, bucketing_config
            )
            msgs.extend(bucketing_msgs)

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


def _apply_min_frequency(
    df: pd.DataFrame, column_name: str, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    """Collapse categories below a minimum frequency threshold into an other bucket.

    Parameters
    ----------
    config : dict
        ``threshold`` : int or float, required
            Minimum count (int >= 1) or proportion (float in (0, 1)) a
            category must reach to be kept.
        ``other_label`` : str, optional, default ``"Other"``
            Label assigned to collapsed categories.
    """
    msgs: list[str] = []

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid 'min_frequency' config for '{column_name}': expected a mapping."
        )

    threshold = config.get("threshold")
    if threshold is None:
        raise ValueError(
            f"'min_frequency' config for '{column_name}' requires a 'threshold' key."
        )

    other_label = config.get("other_label", "Other")

    total = len(df)
    value_counts = df[column_name].value_counts(dropna=False)

    if isinstance(threshold, float) and 0.0 < threshold < 1.0:
        # Proportion mode: convert to absolute count
        min_count = threshold * total
    elif isinstance(threshold, int) and threshold >= 1:
        min_count = float(threshold)
    else:
        raise ValueError(
            f"Invalid threshold for '{column_name}': expected int >= 1 or float in (0, 1), "
            f"got {threshold!r}."
        )

    keep_values = value_counts[value_counts >= min_count].index
    collapsed_count = (value_counts < min_count).sum()

    if collapsed_count == 0:
        msgs.append(
            f"ℹ️ {column_name}: no categories below min_frequency threshold "
            f"(threshold={threshold})."
        )
        return df, msgs

    out_df = df.copy()
    out_df[column_name] = out_df[column_name].where(
        out_df[column_name].isin(keep_values), other_label
    )

    msgs.append(
        f"🧩 {column_name}: collapsed {collapsed_count} rare categories into "
        f"'{other_label}' (threshold={threshold})."
    )
    return out_df, msgs


def _apply_bucketing(
    df: pd.DataFrame, column_name: str, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    """Discretize a numeric column into explicit user-defined bins.

    Parameters
    ----------
    config : dict
        ``edges`` : list of int or float, required
            Monotonically increasing bin boundary values.  N edges produce
            N-1 bins (e.g. ``[0, 25, 50, 100]`` → three bins).
        ``labels`` : list of str, optional
            One label per bin.  When omitted, interval notation is used
            (e.g. ``"(25, 50]"``).
        ``right`` : bool, optional, default ``True``
            Whether intervals are closed on the right.
        ``include_lowest`` : bool, optional, default ``True``
            Whether the leftmost edge is included in the first bin.
    """
    msgs: list[str] = []

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid 'bucketing' config for '{column_name}': expected a mapping."
        )

    edges = config.get("edges")
    if edges is None:
        raise ValueError(
            f"'bucketing' config for '{column_name}' requires an 'edges' key."
        )

    if not isinstance(edges, list) or len(edges) < 2:
        raise ValueError(
            f"'edges' for '{column_name}' must be a list of at least 2 values, "
            f"got {edges!r}."
        )

    labels = config.get("labels")
    right = config.get("right", True)
    include_lowest = config.get("include_lowest", True)

    n_bins = len(edges) - 1
    if labels is not None:
        if not isinstance(labels, list) or len(labels) != n_bins:
            raise ValueError(
                f"'labels' for '{column_name}' must be a list of {n_bins} strings "
                f"(one per bin), got {labels!r}."
            )

    out_df = df.copy()
    out_df[column_name] = pd.cut(
        out_df[column_name],
        bins=edges,
        labels=labels,
        right=right,
        include_lowest=include_lowest,
    )

    label_desc = f"labels={labels}" if labels else f"{n_bins} interval bins"
    msgs.append(
        f"🪣 {column_name}: bucketed into {n_bins} bins (edges={edges}, {label_desc})."
    )
    return out_df, msgs
