#!/usr/bin/env python3
"""
Script Name: embeddings.py
Description: Utility functions for pooling, concatenating, and pivoting AnnData 
             embeddings based on cell and sample metadata.

Usage:
    Imported as a module. Not intended for direct execution.
"""

# Standard Library Imports
from collections.abc import Sequence

# Third-Party Imports
import anndata as ad
import numpy as np
import pandas as pd


def pool_embedding_adata(adata: ad.AnnData,
                         group_by_columns: Sequence[str]) -> ad.AnnData:
    """
    Collapses cell embeddings vertically by taking the mean across specified groups.
    """
    group_label = "_".join(group_by_columns)
    grouped_metadata = adata.obs.groupby(group_by_columns, observed=True)

    # Calculate homogeneity and keep grouping columns as regular columns
    unique_counts_per_group = adata.obs.groupby(
        group_by_columns, observed=True).nunique(dropna=False)
    homogeneous_columns = unique_counts_per_group.columns[(
        unique_counts_per_group.max() == 1).values].tolist()
    safe_columns = list(group_by_columns) + homogeneous_columns

    aggregated_embeddings = []
    aggregated_metadata_rows = []
    aggregated_observation_names = []

    for group_keys, row_indices in grouped_metadata.indices.items():
        # Calculate mean embedding
        group_feature_matrix = adata.X[row_indices]
        mean_embedding_vector = np.asarray(
            group_feature_matrix.mean(axis=0)).flatten()
        aggregated_embeddings.append(mean_embedding_vector)

        # Preserve metadata from the first row of the group safely via dict
        row_dict = adata.obs[safe_columns].iloc[row_indices[0]].to_dict()
        row_dict[group_label + "_count"] = len(row_indices)
        aggregated_metadata_rows.append(row_dict)

        # Construct a safe string index
        if isinstance(group_keys, tuple):
            composite_index_name = "_".join(
                str(key_element) for key_element in group_keys)
            aggregated_observation_names.append(composite_index_name)
        else:
            aggregated_observation_names.append(str(group_keys))

    pooled_adata = ad.AnnData(X=np.array(aggregated_embeddings,
                                         dtype=np.float32),
                              obs=pd.DataFrame(
                                  aggregated_metadata_rows,
                                  index=aggregated_observation_names))

    return pooled_adata


def concatenate_embedding_adata(
        adata: ad.AnnData, group_by_columns: Sequence[str],
        concatenation_axis: str,
        concatenation_order: Sequence[str]) -> ad.AnnData:
    """
    Pivots cell embeddings horizontally in a strict order.
    Generates .var names and drops heterogeneous metadata.
    """
    if not concatenation_axis or not concatenation_order:
        raise ValueError(
            "You must explicitly provide 'concatenation_axis' and 'concatenation_order'."
        )

    grouped_metadata = adata.obs.groupby(group_by_columns, observed=True)
    embedding_dimension = adata.shape[1]

    # 1. Automate Homogeneity Check
    unique_counts_per_group = adata.obs.groupby(
        group_by_columns, observed=True).nunique(dropna=False)
    homogeneous_columns = unique_counts_per_group.columns[(
        unique_counts_per_group.max() == 1).values].tolist()
    safe_columns = list(group_by_columns) + homogeneous_columns

    # 2. Construct feature names for .var
    original_var_names = adata.var_names.astype(str).tolist()
    new_var_names = []
    for target_subgroup_value in concatenation_order:
        new_var_names.extend(
            [f"{target_subgroup_value}_{v}" for v in original_var_names])

    aggregated_embeddings = []
    aggregated_metadata_rows = []
    aggregated_observation_names = []

    for group_keys, row_indices in grouped_metadata.indices.items():
        group_adata_subset = adata[row_indices]
        concatenated_embedding_vector = []

        # Step through the strict order to guarantee dimension stability
        for target_subgroup_value in concatenation_order:
            subgroup_boolean_mask = group_adata_subset.obs[
                concatenation_axis] == target_subgroup_value
            matching_row_count = subgroup_boolean_mask.sum()

            if matching_row_count == 1:
                subgroup_feature_vector = np.asarray(
                    group_adata_subset[subgroup_boolean_mask].X).flatten()
                concatenated_embedding_vector.extend(subgroup_feature_vector)
            elif matching_row_count == 0:
                zero_padding_vector = np.zeros(embedding_dimension,
                                               dtype=np.float32)
                concatenated_embedding_vector.extend(zero_padding_vector)
            else:
                raise ValueError(
                    f"Multiple rows found for {concatenation_axis}={target_subgroup_value} in group {group_keys}. "
                    f"Run 'pool_embedding_adata' first.")

        aggregated_embeddings.append(concatenated_embedding_vector)

        # Preserve safe metadata via dict
        row_dict = adata.obs[safe_columns].iloc[row_indices[0]].to_dict()
        row_dict[concatenation_axis] = "_".join(
            str(val) for val in concatenation_order)
        aggregated_metadata_rows.append(row_dict)

        if isinstance(group_keys, tuple):
            composite_index_name = "_".join(
                str(key_element) for key_element in group_keys)
            aggregated_observation_names.append(composite_index_name)
        else:
            aggregated_observation_names.append(str(group_keys))

    concatenated_adata = ad.AnnData(X=np.array(aggregated_embeddings,
                                               dtype=np.float32),
                                    obs=pd.DataFrame(
                                        aggregated_metadata_rows,
                                        index=aggregated_observation_names),
                                    var=pd.DataFrame(index=new_var_names))

    return concatenated_adata


def pivot_obs_column_to_adata(adata: ad.AnnData,
                              group_by_columns: Sequence[str],
                              concatenation_axis: str,
                              concatenation_order: Sequence[str],
                              target_column: str,
                              fill_value: float = 0.0) -> ad.AnnData:
    """
    Extracts a numeric metadata column, pivots it horizontally, and returns a lightweight AnnData object.
    The .var index is explicitly labeled to combine the subgroup and the target column name.
    """
    grouped_metadata = adata.obs.groupby(group_by_columns, observed=True)

    extracted_rows = []
    observation_names = []

    # 1. Construct feature names for the .var index
    new_var_names = [
        f"{target_subgroup_value}_{target_column}"
        for target_subgroup_value in concatenation_order
    ]

    for group_keys, row_indices in grouped_metadata.indices.items():
        group_obs_subset = adata.obs.iloc[row_indices]
        row_numeric_values = []

        for target_subgroup_value in concatenation_order:
            subgroup_boolean_mask = group_obs_subset[
                concatenation_axis] == target_subgroup_value
            matching_row_count = subgroup_boolean_mask.sum()

            if matching_row_count == 1:
                numeric_val = group_obs_subset.loc[subgroup_boolean_mask,
                                                   target_column].iloc[0]
                row_numeric_values.append(float(numeric_val))
            elif matching_row_count == 0:
                row_numeric_values.append(float(fill_value))
            else:
                raise ValueError(
                    f"Multiple rows found for {concatenation_axis}={target_subgroup_value}. Cannot pivot."
                )

        extracted_rows.append(row_numeric_values)

        # 2. Match the exact string index generation used in the embedding aggregation
        if isinstance(group_keys, tuple):
            composite_index_name = "_".join(
                str(key_element) for key_element in group_keys)
            observation_names.append(composite_index_name)
        else:
            observation_names.append(str(group_keys))

    # 3. Return the lightweight AnnData ready for horizontal concatenation
    return ad.AnnData(X=np.array(extracted_rows, dtype=np.float32),
                      obs=pd.DataFrame(index=observation_names),
                      var=pd.DataFrame(index=new_var_names))
