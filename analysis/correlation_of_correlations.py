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
        for dilution in df['dilution'].unique():
            temp = df[(df['epoch'] == epoch) & (df['dilution'] == dilution)].reset_index(drop=True)
            for layer in ['input', 'hidden', 'output', 'target']:
                # find correlation matrix and remove correlations of a word with itself (i.e. the diagonal)
                layer_activations = temp[layer].apply(pd.Series).astype(float)
                corr_matrix = layer_activations.T.corr().values
                corr_matrix = corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)].reshape(corr_matrix.shape[0], -1)
                temp[f'{layer}_corr_vector'] = corr_matrix.tolist()
            accum.append(temp)
    return pd.concat(accum).reset_index(drop=True)


def calculate_correlation_vectors_new(df, word_sets):
    accum = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if row['orth'] not in word_sets.index:
            continue
        temp = df[(df['epoch'] == row['epoch']) & (df['dilution'] == row['dilution'])
                  & (df['orth'].isin(word_sets.loc[row['orth']]['word_set']))]
        if len(temp) == 0:
            continue

        temp = pd.concat([row.to_frame().T, temp]).reset_index(drop=True)
        for layer in ['input', 'hidden', 'output', 'target']:
            # find correlation matrix and remove correlations of a word with itself (i.e. the diagonal)
            layer_activations = temp[layer].apply(pd.Series).astype(float)
            corr_matrix = layer_activations.T.corr().values
            row[f'{layer}_corr_vector'] = corr_matrix[0][1:]
        accum.append(row)
    return pd.concat(accum, axis=1).T.reset_index(drop=True)


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


def generate_correlation_of_correlations_lineplot(data, dilution=None, error_bars=True,
                                                  hue_order=None, style_order=None):
    """
    Generates the correlation of correlations plot

    Args:
        style_order (list): order for linestyle
        hue_order (list): order for hue colours
        error_bars (bool): whether to plot error bars
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
        sns.lineplot(data=data, x='epoch', y='corr', hue='type', style='corr_type', ci=ci,
                     hue_order=hue_order, style_order=style_order)
    else:
        sns.lineplot(data=data[data['dilution'] == dilution],
                     x='epoch', y='corr', hue='type', style='corr_type', ci=ci,
                     hue_order=hue_order, style_order=style_order)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Correlation')
    if dilution is None:
        ax.set_title('Similarity Correlations (Plaut Figure 18)')
    else:
        ax.set_title(f'Similarity Correlations (Plaut Figure 18) - Dilution Level: {dilution}')
    plt.show()


def generate_correlation_of_correlations_barplot(data, epoch, dilution=None, ylim=(0.6, 0.9),
                                                 order=None, hue_order=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    if dilution is None:
        sns.barplot(data=data[data['epoch'] == epoch], ax=ax, ci=68,
                    x='corr_type', y="corr", hue='type', errwidth=1, capsize=.1, order=order, hue_order=hue_order)
    else:
        sns.barplot(data=data[(data['epoch'] == epoch) & (data['dilution'] == dilution)], ax=ax, ci=68,
                    x='corr_type', y="corr", hue='type', errwidth=1, capsize=.1, order=order, hue_order=hue_order)
    ax.set_xlabel('Correlation Type')
    ax.set_ylabel('Correlation')
    ax.set_ylim(*ylim)
    if dilution is None:
        ax.set_title('Similarity Correlations (Plaut Figure 18)')
    else:
        ax.set_title(f'Similarity Correlations (Plaut Figure 18) - Dilution Level: {dilution}')
    plt.show()
