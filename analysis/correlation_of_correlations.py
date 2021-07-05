"""
Correlation of Correlations

=== SUMMARY ===
Description     : This file contains some helper functions for any "correlation of correlation" (Plaut Fig18) analysis
Notes           : The functions here are modified from hidden_similarity_lens.py for use in Jupyter Notebook analysis
Date Created    : June 26, 2021
Last Updated    : June 26, 2021

=== UPDATE NOTES ===
 > June 26, 2021
    - file created
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm


def calculate_correlation_vectors(df):
    """
    Calculate the correlation between a word and all other words in the given layer

    Args:
        df (pd.DataFrame): the input, hidden and output activations, and target activation

    Returns:
        pd.DataFrame
    """
    accum = []
    for epoch in tqdm(df['epoch'].unique(), total=df['epoch'].nunique()):
        for dilution in range(1, 4):
            temp = df[(df['epoch'] == epoch) & (df['dilution'] == dilution)].reset_index(drop=True)
            for layer in ['input', 'hidden', 'output', 'target']:
                corr_df = temp[layer].apply(pd.Series).T.corr()
                temp[f'{layer}_corr_vector'] = corr_df.values.tolist()
                # temp[f'{layer}_corr_vector'] = temp.apply(
                #     lambda row: row[f'{layer}_corr_vector'][:row.name] + row[f'{layer}_corr_vector'][row.name + 1:],
                #     axis=1)
            accum.append(temp)

    return pd.concat(accum).reset_index(drop=True)


def calculate_cross_layer_correlation(df):
    """
    Calculates the "correlation of correlations" - i.e. the correlation of two corr_vectors produced by the
    calculate_correlation_vectors function

    Args:
        df (pd.DataFrame): the correlation vectors

    Returns:
        pd.DataFrame
    """

    df['orth_hidden_corr'] = df.apply(lambda row: np.corrcoef(row['input_corr_vector'],
                                                              row['hidden_corr_vector'])[0, 1], axis=1)
    df['hidden_phon_corr'] = df.apply(lambda row: np.corrcoef(row['hidden_corr_vector'],
                                                              row['output_corr_vector'])[0, 1], axis=1)
    df['orth_phon_corr'] = df.apply(lambda row: np.corrcoef(row['input_corr_vector'],
                                                            row['output_corr_vector'])[0, 1], axis=1)
    df['orth_target_corr'] = df.apply(lambda row: np.corrcoef(row['input_corr_vector'],
                                                              row['target_corr_vector'])[0, 1], axis=1)
    return df


def generate_correlation_of_correlations_lineplot(data, dilution=None, error_bars=True):
    """
    Generates the correlation of correlations plot

    Args:
        data (pd.DataFrame): dataframe containing epoch, correlation, word type
        dilution (int): dilution level

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=(15, 5))
    if error_bars:
        ci = 68
    else:
        ci = None
    if dilution is None:
        sns.lineplot(data=data, x='epoch', y='corr', hue='type', style='corr_type', ci=ci)
    else:
        sns.lineplot(data=data[data['dilution'] == dilution], x='epoch', y='corr', hue='type', style='corr_type', ci=ci)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Correlation')
    if dilution is None:
        ax.set_title('Similarity Correlations (Plaut Figure 18)')
    else:
        ax.set_title(f'Similarity Correlations (Plaut Figure 18) - Dilution Level: {dilution}')
    plt.show()


def generate_correlation_of_correlations_barplot(data, epoch, dilution=None, ylim=(0.6, 0.9)):
    fig, ax = plt.subplots(figsize=(6, 5))
    if dilution is None:
        sns.barplot(data=data[data['epoch'] == epoch], ax=ax,
                    x='corr_type', y="corr", hue='type', errwidth=1, capsize=.1)
    else:
        sns.barplot(data=data[(data['epoch'] == epoch) & (data['dilution'] == dilution)], ax=ax,
                    x='corr_type', y="corr", hue='type', errwidth=1, capsize=.1)
    ax.set_xlabel('Correlation Type')
    ax.set_ylabel('Correlation')
    ax.set_ylim(*ylim)
    if dilution is None:
        ax.set_title('Similarity Correlations (Plaut Figure 18)')
    else:
        ax.set_title(f'Similarity Correlations (Plaut Figure 18) - Dilution Level: {dilution}')
    plt.show()
