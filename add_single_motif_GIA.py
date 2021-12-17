import sys
import tensorflow as tf
import h5py, os, yaml
import umap.umap_ as umap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import seaborn as sns
from scipy import stats
import pandas as pd
import subprocess
from scipy.stats import pearsonr
from tqdm import tqdm
import glob
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator
import tfr_evaluate, util
from test_to_bw_fast import read_model, get_config
import explain
import embed
import metrics
import quant_GIA
from optparse import OptionParser
import itertools

# inputs: cell line, motifs
run_path = 'paper_runs/new_models/32_res/run-20211023_095131-w6okxt01'
model, bin_size = read_model(run_path, compile_model=False)

testset, targets = tfr_evaluate.collect_whole_testset(coords=True)


low_C, low_X, low_Y = embed.threshold_cell_line_testset(testset, cell_line, more_than=1, less_than=2)
# [('none', low_X), ('dinuc', high_X), ('random', high_X)]

# for each background: low none, dinuc or random high for all cell lines
background_type, X_set = ('none', low_X)
gi = GlobalImportance(model)
gi.set_null_model(background, base_sequence=X_set, num_sample=1000)
# for each motif in the tandem of motifs
# for motif in tandem_motifs_to_add:
# select the best flanks based on where the dots are in the pattern
generate_flanks()
best_flank = test_flanks()
# slide across half sec to determine best position

gi.positional_bias(best_flank, positions=range(0,2048//2,2))


# if 2 motifs fix one and shift other

def test_flanks(gi, motif, position=1024):
    diff_scores = gi.embed_predict_quant_effect([(motif_with_flanking_nucls, position)])
    all_scores_per_motif=(diff_scores).mean(axis=0).mean(axis=0)
    all_scores.append(all_scores_per_motif)


def generate_flanks(motif_pattern):
    dot_positions = np.argwhere(np.array(list(motif_pattern))=='.').flatten()
    kmer_size = len(dot_positions)
    kmers = ["".join(p) for p in itertools.product(list('ACGT'), repeat=kmer_size)]
    all_motifs = []
    for kmer in tqdm(kmers):

        motif_with_flanking_nucls = list(motif_pattern)
        for p, pos in enumerate(dot_positions):
            motif_with_flanking_nucls[pos] = kmer[p]
        motif_with_flanking_nucls = ''.join(motif_with_flanking_nucls)
        all_motifs.append(motif_with_flanking_nucls)
    return all_motifs
