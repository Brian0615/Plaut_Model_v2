"""
helpers.py

=== SUMMARY ===
Description     : Miscellaneous helper functions
Date Created    : May 03, 2020
Last Updated    : August 4, 2020

=== DETAILED DESCRIPTION ===
 > Changes from v1:
    - make_folder function renamed to create_simulation_folder, and now forces a folder label to be given

=== UPDATE NOTES ===
 > August 4, 2020
    - add get_graphemes and get_phonemes functions from simulator/dataset.py
 > July 27, 2020
    - add function to create analysis folder
 > July 26, 2020
    - add series folder
 > July 19, 2020
    - formatting changes
    - fix too broad except clause
 > May 24, 2020
    - Move helper functions from plaut_model.py into this file
    - update string format, filepath, import statements
 > May 08, 2020
    - update create_simulation_folder to create new folder with suffix if folder already exists
      and user does not want to delete the original folder
 > May 03, 2020
    - file created
"""

import os
import shutil
from common.constants import VectorMapping


def create_simulation_folder(dir_label, series=False):
    # create a new folder for every run
    if series:
        series_dir = "results/" + dir_label.split('-')[0] + "-" + dir_label.split('-')[-1]
        try:
            os.mkdir(series_dir)
        except FileExistsError:
            pass
        rootdir = series_dir + "/" + dir_label
    else:
        rootdir = "results/" + dir_label

    try:
        os.mkdir(rootdir)
    except FileExistsError:
        while True:
            replace = input(f"{rootdir} already exists. Replace? (Y/N)")
            if replace in ['Y', 'N', 'y', 'n']:
                break
        if replace in ['Y', 'y']:
            print("Overwriting existing directory...")
            shutil.rmtree(rootdir)
            os.mkdir(rootdir)
        else:
            i = 1
            while True:
                try:
                    new_rootdir = f"{rootdir}_{i}"
                    os.mkdir(new_rootdir)
                    break
                except FileExistsError:
                    i += 1
            rootdir = new_rootdir
            dir_label = f"{dir_label}_{i}"
    for subdir in ["Training Loss", "Training Accuracy", "Anchor Accuracy", "Probe Accuracy"]:
        os.mkdir(rootdir + "/" + subdir)
    return rootdir, dir_label


def create_analysis_folder(results_dir, analysis_type):
    rootdir = results_dir + "/" + analysis_type
    try:
        os.mkdir(rootdir)
    except FileExistsError:
        while True:
            replace = input(f"{rootdir} already exists. Replace? (Y/N)")
            if replace in ['Y', 'N', 'y', 'n']:
                break
        if replace in ['Y', 'y']:
            print("Overwriting existing directory...")
            shutil.rmtree(rootdir)
            os.mkdir(rootdir)
        else:
            i = 1
            while True:
                try:
                    new_rootdir = f"{rootdir}_{i}"
                    os.mkdir(new_rootdir)
                    break
                except FileExistsError:
                    i += 1
            rootdir = new_rootdir

    return rootdir

def get_graphemes(word):
    word = str(word).upper()  # convert all text to capitals first
    if word == "NAN":  # the word null automatically gets imported as "NaN" in dataframe, so fix that
        word = "NULL"

    grapheme_onset = VectorMapping.grapheme_onset
    grapheme_vowel = VectorMapping.grapheme_vowel
    grapheme_codas = VectorMapping.grapheme_codas

    # initialize vectors to zero
    onset = [0] * len(grapheme_onset)
    vowel = [0] * len(grapheme_vowel)
    codas = [0] * len(grapheme_codas)

    # for onset: essentially "turn on" corresponding slots for onsets until a vowel is reached
    i = 0
    for i in range(len(word)):
        if word[i] in grapheme_vowel:  # vowel found, move on
            if not (i == 0 and word[i] == 'Y'):
                break
        if word[i] in grapheme_onset:  # single-letter grapheme found
            onset[grapheme_onset.index(word[i])] = 1
        if word[i:i + 2] in grapheme_onset:  # double-letter grapheme found
            onset[grapheme_onset.index(word[i:i + 2])] = 1

    # for vowels
    vowel[grapheme_vowel.index(word[i])] = 1
    if i + 1 < len(word):  # check for double-vowel
        if word[i + 1] in grapheme_vowel:
            vowel[grapheme_vowel.index(word[i + 1])] = 1
        if word[i:i + 2] in grapheme_vowel:
            vowel[grapheme_vowel.index(word[i:i + 2])] = 1
        # if double-letter vowel found, increment i one more time
        if word[i + 1] in grapheme_vowel or word[i:i + 2] in grapheme_vowel:
            i += 1

    # for codas
    for j in range(i + 1, len(word)):
        if word[j] in grapheme_codas:  # check for single-letter coda
            codas[grapheme_codas.index(word[j])] = 1
        if word[j:j + 2] in grapheme_codas:  # check for double-letter coda
            codas[grapheme_codas.index(word[j:j + 2])] = 1
        if word[j:j + 3] in grapheme_codas:  # check for triple-letter coda
            codas[grapheme_codas.index(word[j:j + 3])] = 1

    # combine and return
    return onset + vowel + codas


# similar idea to graphemes; refer to above for comments
def get_phonemes(phon):
    phoneme_onset = VectorMapping.phoneme_onset
    phoneme_vowel = VectorMapping.phoneme_vowel
    phoneme_codas = VectorMapping.phoneme_codas

    phon = phon.strip('/')
    onset = [0] * len(phoneme_onset)
    vowel = [0] * len(phoneme_vowel)
    codas = [0] * len(phoneme_codas)

    i, j, k = 0, 0, 0
    for i in range(len(phon)):
        if phon[i] in phoneme_vowel:
            break
        if phon[i] in phoneme_onset:
            onset[phoneme_onset.index(phon[i])] = 1

    for j in range(i, len(phon)):
        if phon[j] in phoneme_codas:
            break
        if phon[j] in phoneme_vowel:
            vowel[phoneme_vowel.index(phon[j])] = 1

    for k in range(j, len(phon)):
        if phon[k] in phoneme_codas:
            codas[phoneme_codas.index(phon[k])] = 1
        if phon[k:k + 2] in phoneme_codas:
            codas[phoneme_codas.index(phon[k:k + 2])] = 1

    return onset + vowel + codas