from typing import List, Mapping, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

from src.visualization.visualization_helpers import form_path


def plot_distribution_for_nominal_features(df: pd.DataFrame, features: List[str], n_cols: int, n_rows: int, save_filename: Optional[str] = None):
    # Keeping the giant vertical canvas space you created
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 10))

    if n_rows == 1:
        axes = [axes]

    for i, col in enumerate(features):
        if col in df.columns:
            sns.countplot(
                y=col,  # Switched back to y to let labels sit perfectly flat on the left
                data=df,
                ax=axes[i],
                palette="ch:.25",
                order=df[col].value_counts().index,
                # hue="Target"
            )
            axes[i].set_title(f"Distribution of {col}", fontsize=16, fontweight='bold', pad=15)
            axes[i].set_ylabel("")  # Cleaning up redundant vertical title
            axes[i].set_xlabel("Count", fontsize=12)

            # Smart Text Wrapping for the y-axis ticks
            labels = [item.get_text() for item in axes[i].get_yticklabels()]
            # Wraps text at 40 characters for comfortable left-side reading
            wrapped_labels = ['\n'.join(textwrap.wrap(l, 40)) for l in labels]

            axes[i].set_yticklabels(wrapped_labels, fontsize=10)

    # Structural spacing between separate figure grids
    plt.subplots_adjust(hspace=0.5)

    if save_filename:
        path_file = form_path(save_filename)
        plt.savefig(
            path_file,
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )

    plt.show()


def plot_distribution_for_ordinal_features(df: pd.DataFrame, features: List[str], order_maps: Mapping[str, List[str]], n_cols: int, n_rows: int, save_filename: Optional[str] = None):
    # Setup canvas space
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 8))

    if n_rows == 1:
        axes = [axes]

    for i, col in enumerate(features):
        if col in df.columns:
            # Filter the defined order list to only include categories actually present in the data
            present_categories = df[col].dropna().unique()
            plot_order = [cat for cat in order_maps[col] if cat in present_categories]

            sns.countplot(
                y=col,
                data=df,
                ax=axes[i],
                palette="flare",  # Switched palette slightly to visually differentiate ordinal vs nominal
                order=plot_order,  # Keeps the structural sequence instead of sorting by size
                # hue="Target"
            )

            axes[i].set_title(f"Logical Distribution of {col}", fontsize=16, fontweight='bold', pad=15)
            axes[i].set_ylabel("")
            axes[i].set_xlabel("Count", fontsize=12)

            # Wrap the text labels nicely on the left side
            labels = [item.get_text() for item in axes[i].get_yticklabels()]
            wrapped_labels = ['\n'.join(textwrap.wrap(l, 40)) for l in labels]

            axes[i].set_yticklabels(wrapped_labels, fontsize=10)

    plt.subplots_adjust(hspace=0.5)

    if save_filename:
        path_file = form_path(save_filename)
        plt.savefig(
            path_file,
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )

    plt.show()


def plot_distribution_for_binary_features(df: pd.DataFrame, features: List[str], label_maps: Mapping[str, Mapping[int, str]], n_cols: int, save_filename: Optional[str] = None):
    num_features = len(features)
    n_rows = (num_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4.5))
    axes = axes.flatten()

    for i, col in enumerate(features):
        if col in df.columns:
            # Get the unique values present in the column and sort them (usually [0, 1])
            unique_vals = sorted(df[col].dropna().unique())

            # Build the specific text labels for the categories present in this column
            display_labels = [label_maps[col].get(val, str(val)) for val in unique_vals]

            sns.countplot(
                x=col,
                data=df,
                ax=axes[i],
                palette="Set2",
                # hue="Target"
            )

            axes[i].set_title(f"Distribution of {col}", fontsize=13, fontweight='bold', pad=10)
            axes[i].set_xlabel("")  # Removed generic "0 / 1" label since ticks are descriptive now
            axes[i].set_ylabel("Count", fontsize=10)

            # Apply the true text labels safely onto the x-axis ticks
            axes[i].set_xticks(range(len(unique_vals)))
            axes[i].set_xticklabels(display_labels, fontsize=11)
            axes[i].tick_params(axis='y', labelsize=9)

    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if save_filename:
        path_file = form_path(save_filename)
        plt.savefig(
            path_file,
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )

    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, save_filename: Optional[str] = None, title: Optional[str] = None):
    plt.figure(figsize=(25, 20))
    sns.heatmap(df.corr(),
                annot=True,
                linewidths=0.5,
                fmt=".2f",
                cmap="YlGnBu"
                )

    if save_filename:
        if title:
            plt.title(title)
        path_file = form_path(save_filename)
        plt.savefig(path_file)

    plt.show()


