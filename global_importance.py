from tensorflow import keras
from tensorflow.keras import backend as K
from scipy import stats
import numpy as np
import itertools
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator
import seaborn as sns

# from dinuc_shuffle import dinuc_shuffle



class GlobalImportance():
    """Class that performs GIA experiments."""
    def __init__(self, model, alphabet='ACGU'):
        self.model = model
        self.alphabet = alphabet
        self.x_null = None
        self.x_null_index = None
        self.embedded_predictions = {}


    def set_null_model(self, null_model, base_sequence, quant=False, num_sample=1000, binding_scores=None, seed=None):
        """use model-based approach to set the null sequences"""
        self.x_null, self.null_sample_idx = generate_null_sequence_set(null_model, base_sequence, num_sample, binding_scores, seed)
        self.x_null_index = np.argmax(self.x_null, axis=2)
        self.predict_null(quant=quant)


    def set_x_null(self, x_null):
        """set the null sequences"""
        self.x_null = x_null
        self.x_null_index = np.argmax(x_null, axis=2)
        self.predict_null()


    def filter_null(self, low=10, high=90, num_sample=1000):
        """ remove sequences that yield extremum predictions"""
        high = np.percentile(self.null_scores, high)
        low = np.percentile(self.null_scores, low)
        index = np.where((self.null_scores < high)&(self.null_scores > low))[0]
        self.set_x_null(self.x_null[index][:num_sample])
        self.predict_null()


    def predict_null(self, class_index=None, quant=False):
        """perform GIA on null sequences"""
        if quant:
            self.null_profiles = self.model.predict(self.x_null)
        else:
            self.null_scores = self.model.predict(self.x_null)[:, class_index]
            self.mean_null_score = np.mean(self.null_scores)


    def embed_patterns(self, patterns):
        """embed patterns in null sequences"""
        if not isinstance(patterns, list):
            patterns = [patterns]

        x_index = np.copy(self.x_null_index)
        for pattern, position in patterns:

            # convert pattern to categorical representation
            pattern_index = np.array([self.alphabet.index(i) for i in pattern])

            # embed pattern
            x_index[:,position:position+len(pattern)] = pattern_index

        # convert to categorical representation to one-hot
        one_hot = np.zeros((len(x_index), len(x_index[0]), len(self.alphabet)))
        for n, x in enumerate(x_index):
            for l, a in enumerate(x):
                one_hot[n,l,a] = 1.0

        return one_hot


    def set_hairpin_null(self, stem_left=7, stem_right=23, stem_size=9):
        """create a hairpin for the null sequences"""
        one_hot = np.copy(self.x_null)
        stem_left_end = stem_left + stem_size
        stem_right_end = stem_right + stem_size
        rc = one_hot[:,stem_left:stem_left_end,:]
        rc = rc[:,:,::-1]
        rc = rc[:,::-1,:]
        one_hot[:,stem_right:stem_right_end,:] = rc
        self.set_x_null(one_hot)


    def embed_pattern_hairpin(self, patterns, stem_left=7, stem_right=23, stem_size=9):
        """embed pattern within a hairpin for the null sequences"""

        # set the null to be a stem-loop
        self.set_hairpin_null(stem_left=7, stem_right=23, stem_size=9)

        # embed the pattern
        one_hot = self.embed_patterns(patterns)

        # fix the step
        stem_left_end = stem_left + stem_size
        stem_right_end = stem_right + stem_size
        rc = one_hot[:,stem_left:stem_left_end,:]
        rc = rc[:,:,::-1]
        rc = rc[:,::-1,:]
        one_hot[:,stem_right:stem_right_end,:] = rc

        return  one_hot


    def embed_predict_effect(self, patterns, class_index=0):
        """embed pattern in null sequences and get their predictions"""
        one_hot = self.embed_patterns(patterns)
        return self.model.predict(one_hot)[:, class_index] - self.null_scores

    def embed_predict_quant_effect(self, patterns, class_index=0):
        """embed pattern in null sequences and get their predictions"""
        one_hot = self.embed_patterns(patterns)
        pattern_label = ' & '.join(['{} at {}'.format(m, str(p)) for m,p in patterns])
        self.embedded_predictions[pattern_label] = self.model.predict(one_hot)
        assert self.embedded_predictions[pattern_label].shape == self.null_profiles.shape
        return self.embedded_predictions[pattern_label] - self.null_profiles

    def predict_effect(self, one_hot, class_index=0):
        """Measure effect size of sequences versus null sequences"""
        predictions = self.model.predict(one_hot)[:, class_index]
        return predictions - self.null_scores


    def optimal_kmer(self, kmer_size=7, position=17, class_index=0):
        """GIA to find optimal k-mers"""

        # generate all kmers
        kmers = ["".join(p) for p in itertools.product(list(self.alphabet), repeat=kmer_size)]

        # score each kmer
        mean_scores = []
        for i, kmer in enumerate(kmers):
            if np.mod(i+1,500) == 0:
                print("%d out of %d"%(i+1, len(kmers)))

            effect = self.embed_predict_effect((kmer, position), class_index)
            mean_scores.append(np.mean(effect))

        kmers = np.array(kmers)
        mean_scores = np.array(mean_scores)

        # sort by highest prediction
        sort_index = np.argsort(mean_scores)[::-1]

        return kmers[sort_index], mean_scores[sort_index]


    def kmer_mutagenesis(self, kmer='UGCAUG', position=17, class_index=0):
        """GIA mutagenesis of a k-mer"""

        # get wt score
        wt_score = np.mean(self.embed_predict_effect((kmer, position), class_index))

        # score each mutation
        L = len(kmer)
        A = len(self.alphabet)
        mean_scores = np.zeros((L, A))
        for l in range(L):
            for a in range(A):
                if kmer[l] == self.alphabet[a]:
                    mean_scores[l,a] = wt_score

                else:
                    # introduce mutation
                    mut_kmer = list(kmer)
                    mut_kmer[l] = self.alphabet[a]
                    mut_kmer = "".join(mut_kmer)

                    # score mutant
                    mean_scores[l,a]  = np.mean(self.embed_predict_effect((mut_kmer, position), class_index))

        return mean_scores



    def positional_bias(self, motif='UGCAUG', positions=[2, 12, 23, 33], class_index=0):
        """GIA to find positional bias"""

        # loop over positions and measure effect size of intervention
        all_scores = []
        for position in positions:
            all_scores.append(self.embed_predict_effect((motif, position), class_index))

        return np.array(all_scores)



    def multiple_sites(self, motif='UGCAUG', positions=[17, 10, 25, 3], class_index=0):
        """GIA to find relation with multiple binding sites"""

        # loop over positions and measure effect size of intervention
        all_scores = []
        for i, position in enumerate(positions):

            # embed motif multiple times
            interventions = []
            for j in range(i+1):
                interventions.append((motif, positions[j]))

            all_scores.append(self.embed_predict_effect(interventions, class_index))

        return np.array(all_scores)


    def gc_bias(self, motif='UGCAUG', motif_position=17,
                gc_motif='GCGCGC', gc_positions=[34, 2], class_index=0):
        """GIA to find GC-bias"""

        all_scores = []


        # background sequence with gc-bias on right side
        all_scores.append(self.embed_predict_effect((gc_motif, gc_positions[0]), class_index))

        # background sequence with motif at center
        all_scores.append(self.embed_predict_effect((motif, motif_position), class_index))

        # create interventions for gc bias
        for position in gc_positions:

            interventions = [(motif, motif_position), (gc_motif, position)]
            all_scores.append(self.embed_predict_effect(interventions, class_index))

        return np.array(all_scores)


#-------------------------------------------------------------------------------------
# Null sequence models
#-------------------------------------------------------------------------------------


def generate_null_sequence_set(null_model, base_sequence, num_sample=1000 , binding_scores=None, seed=None):
    if null_model == 'random':    return generate_shuffled_set(base_sequence, num_sample)
    if null_model == 'profile':   return generate_profile_set(base_sequence, num_sample)
    if null_model == 'dinuc':     return generate_dinucleotide_shuffled_set(base_sequence, num_sample)
    if null_model == 'quartile1': return generate_quartile_set(base_sequence, num_sample, binding_scores, quartile=1)
    if null_model == 'quartile2': return generate_quartile_set(base_sequence, num_sample, binding_scores, quartile=2)
    if null_model == 'quartile3': return generate_quartile_set(base_sequence, num_sample, binding_scores, quartile=3)
    if null_model == 'quartile4': return generate_quartile_set(base_sequence, num_sample, binding_scores, quartile=4)
    if null_model == 'none':
        if seed:
            np.random.seed(seed)
            print('seed set!')
        idx = np.random.choice(base_sequence.shape[0], num_sample)

        return base_sequence[idx], idx
    else: print ('null_model name not recognized.')


def generate_profile_set(base_sequence, num_sample):
    # set null sequence model
    seq_model = np.mean(np.squeeze(base_sequence), axis=0)
    seq_model /= np.sum(seq_model, axis=1, keepdims=True)

    # sequence length
    L = seq_model.shape[0]

    x_null = np.zeros((num_sample, L, 4))
    for n in range(num_sample):

        # generate uniform random number for each nucleotide in sequence
        Z = np.random.uniform(0,1,L)

        # calculate cumulative sum of the probabilities
        cum_prob = seq_model.cumsum(axis=1)

        # find bin that matches random number for each position
        for l in range(L):
            index = [j for j in range(4) if Z[l] < cum_prob[l,j]][0]
            x_null[n,l,index] = 1
    return x_null


def generate_shuffled_set(base_sequence, num_sample):
    # take a random subset of base_sequence
    shuffle = np.random.permutation(len(base_sequence))
    x_null = base_sequence[shuffle[:num_sample]]

    # shuffle nucleotides
    [np.random.shuffle(x) for x in x_null]
    return x_null


def generate_dinucleotide_shuffled_set(base_sequence, num_sample):

    # take a random subset of base_sequence
    shuffle = np.random.permutation(len(base_sequence))
    x_null = base_sequence[shuffle[:num_sample]]

    # shuffle dinucleotides
    for j, seq in enumerate(x_null):
        x_null[j] = dinuc_shuffle(seq)
    return x_null


def generate_quartile_set(base_sequence, num_sample, binding_scores, quartile):
    # sort sequences by the binding score (descending order)
    sort_index = np.argsort(binding_scores[:,0])[::-1]
    base_sequence = base_sequence[sort_index]

    # set quartile indices
    L = len(base_sequence)
    L0, L1, L2, L3, L4 = [0, int(L/4), int(L*2/4), int(L*3/4), L]

    # pick the quartile:
    if (quartile==1): base_sequence = base_sequence[L0:L1]
    if (quartile==2): base_sequence = base_sequence[L1:L2]
    if (quartile==3): base_sequence = base_sequence[L2:L3]
    if (quartile==4): base_sequence = base_sequence[L3:L4]

    # now shuffle the sequences
    shuffle = np.random.permutation(len(base_sequence))

    # take a smaller sample of size num_sample
    return base_sequence[shuffle[:num_sample]]





#-------------------------------------------------------------------------------------
# Useful functions
#-------------------------------------------------------------------------------------

def boxplot_with_test(data, x, y, pairs):
    plotting_parameters = {
                            'data':    data,
                            'x':       x,
                            'y':       y}
    pvalues = [mannwhitneyu(data[data[x]==pair[0]][y],
                            data[data[x]==pair[1]][y]).pvalue for pair in pairs]
    ax = sns.boxplot(**plotting_parameters)
    # Add annotations
    annotator = Annotator(ax, pairs, **plotting_parameters)
    annotator.set_pvalues(pvalues)
    annotator.annotate();

def get_saliency_values(seq, model, class_index):
    explainer = explain.Explainer(model, class_index=class_index)
    x = np.expand_dims(seq, axis=0)
    saliency_scores = explainer.saliency_maps(x)
    grad_times_input = np.sum(saliency_scores[0]*seq, axis=1)
    return grad_times_input

def get_multiple_saliency_values(seqs, model, class_index):
    explainer = explain.Explainer(model, class_index=class_index)
    saliency_scores = explainer.saliency_maps(seqs)
    grad_times_input = np.sum(saliency_scores*seqs, axis=-1)
    return grad_times_input

def pearsonr_scores(y_true, y_pred, mask_value=None):
    corr = []
    for i in range(y_true.shape[1]):
        if mask_value:
            index = np.where(y_true[:,i] != mask_value)[0]
            corr.append(stats.pearsonr(y_true[index,i], y_pred[index,i])[0])
        else:
            corr.append(stats.pearsonr(y_true[:,i], y_pred[:,i])[0])
    return np.array(corr)

def dinuc_shuffle(seq, num_shufs=None, rng=None):
    """
    Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.
    Arguments:
        `seq`: either a string of length L, or an L x D NumPy array of one-hot
            encodings
        `num_shufs`: the number of shuffles to create, N; if unspecified, only
            one shuffle will be created
        `rng`: a NumPy RandomState object, to use for performing shuffles
    If `seq` is a string, returns a list of N strings of length L, each one
    being a shuffled version of `seq`. If `seq` is a 2D NumPy array, then the
    result is an N x L x D NumPy array of shuffled versions of `seq`, also
    one-hot encoded. If `num_shufs` is not specified, then the first dimension
    of N will not be present (i.e. a single string will be returned, or an L x D
    array).
    """
    if type(seq) is str:
        arr = string_to_char_array(seq)
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = one_hot_to_tokens(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")

    if not rng:
        rng = np.random.RandomState()

    # Get the set of all characters, and a mapping of which positions have which
    # characters; use `tokens`, which are integer representations of the
    # original characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token

    if type(seq) is str:
        all_results = []
    else:
        all_results = np.empty(
            (num_shufs if num_shufs else 1, seq_len, one_hot_dim),
            dtype=seq.dtype
        )

    for i in range(num_shufs if num_shufs else 1):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)

        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        if type(seq) is str:
            all_results.append(char_array_to_string(chars[result]))
        else:
            all_results[i] = tokens_to_one_hot(chars[result], one_hot_dim)
    return all_results if num_shufs else all_results[0]



def string_to_char_array(seq):
    """
    Converts an ASCII string to a NumPy array of byte-long ASCII codes.
    e.g. "ACGT" becomes [65, 67, 71, 84].
    """
    return np.frombuffer(bytearray(seq, "utf8"), dtype=np.int8)


def char_array_to_string(arr):
    """
    Converts a NumPy array of byte-long ASCII codes into an ASCII string.
    e.g. [65, 67, 71, 84] becomes "ACGT".
    """
    return arr.tostring().decode("ascii")


def one_hot_to_tokens(one_hot):
    """
    Converts an L x D one-hot encoding into an L-vector of integers in the range
    [0, D], where the token D is used when the one-hot encoding is all 0. This
    assumes that the one-hot encoding is well-formed, with at most one 1 in each
    column (and 0s elsewhere).
    """
    tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens


def tokens_to_one_hot(tokens, one_hot_dim):
    """
    Converts an L-vector of integers in the range [0, D] to an L x D one-hot
    encoding. The value `D` must be provided as `one_hot_dim`. A token of D
    means the one-hot encoding is all 0s.
    """
    identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
    return identity[tokens]


#-------------------------------------------------------------------------------------
# functions to find a motif in a sequence
#-------------------------------------------------------------------------------------
def find_motif_indices(motif_pattern, str_seq):
    '''Find all str motif start positions in a sequence str'''
    iter = re.finditer(motif_pattern, str_seq)
    return [m.start(0) for m in iter]

def find_max_saliency_ind(indices, saliency_values):
    '''find motif instance closest to the max saliency value'''
    max_point = np.argmax(saliency_values)
    if len(indices)>0:
        return [indices[np.abs(indices-max_point).argmin()]]
    else:
        return []

def filter_indices_in_saliency_peak(indices, saliency_values, window=300):
    '''filter motifs within a window around the max saliency'''
    max_point = np.argmax(saliency_values)
    if len(indices)>0:
        return list(np.array(indices)[(np.abs(indices-max_point)<window/2)])
    else:
        return []

def select_indices(motif_pattern, str_seq, saliency_values=None,
                   max_only=False, filter_window=False):
    '''select indices according to filtering criteria'''
    indices = find_motif_indices(motif_pattern, str_seq)
    if max_only:
        return find_max_saliency_ind(indices, saliency_values)
    elif filter_window:
        return filter_indices_in_saliency_peak(indices, saliency_values, filter_window)
    else: # find all
        return indices


def find_multiple_motifs(motif_pattern_list, str_seq, saliency_values=None,
                        max_only=False, filter_window=False):
    '''find indices of multiple motifs in a single sequence'''
    motifs_and_indices = {}
    for motif_pattern in motif_pattern_list:
        chosen_ind = select_indices(motif_pattern, str_seq,
                                    saliency_values,
                                    max_only, filter_window )
        motifs_and_indices[motif_pattern] = chosen_ind
    return motifs_and_indices

def find_seqs_with_motif(ind_subset, motif_pattern, all_X, model, cell_line):
    del_ind = []
    for i,ind in enumerate(ind_subset):
        str_seq = ''.join(onehot_to_str(all_X[ind,:,:]))
        grad_times_input = get_saliency_values(all_X[ind,:,:], model, cell_line)
        motif_d = find_multiple_motifs([motif_pattern], str_seq,
                             saliency_values=grad_times_input, filter_window=256)
        if len(motif_d[motif_pattern])==0:
            del_ind.append(i)
    ind_subset = np.delete(ind_subset, del_ind)
    return ind_subset
#-------------------------------------------------------------------------------------
# functions to remove or randomize a motif
#-------------------------------------------------------------------------------------

def remove_motif_from_seq(motifs_and_indices, selected_seq):
    modified_seq = selected_seq.copy()
    for motif_pattern, motif_start_indices in motifs_and_indices.items():
        for motif_start in motif_start_indices:
            motif_end = motif_start+len(motif_pattern)
            empty_pattern = np.zeros_like(selected_seq[motif_start:motif_end])+0.25

            modified_seq[motif_start:motif_end] = empty_pattern
    return modified_seq

def randomize_motif_in_seq(motifs_and_indices, selected_seq, n_occlusions=25):
    modified_seqs = []
    for i in range(n_occlusions):
        modified_seq = selected_seq.copy()
        for motif_pattern, motif_start_indices in motifs_and_indices.items():
            for motif_start in motif_start_indices:
                random_pattern = np.array([[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]][np.random.randint(4)] for i in range(len(motif_pattern))])
                modified_seq[motif_start:motif_start+len(motif_pattern)] = random_pattern
        modified_seqs.append(modified_seq)
    return np.array(modified_seqs)

def randomize_multiple_seqs(onehot_seqs, motif_pattern, model, cell_line):
    seqs_with_motif = []
    seqs_removed_motifs = []
    saliency_all_seqs = get_multiple_saliency_values(onehot_seqs, model, cell_line)
    for o, onehot_seq in enumerate(onehot_seqs):
        str_seq = ''.join(onehot_to_str(onehot_seq))
        motifs_and_indices = (find_multiple_motifs([motif_pattern], str_seq,
                             saliency_values=saliency_all_seqs[o], filter_window=256))
        if motifs_and_indices[motif_pattern]:
            seqs_with_motif.append(onehot_seq.copy())
            seqs_removed_motifs.append(randomize_motif_in_seq(motifs_and_indices,
                                                              onehot_seq))
    return (seqs_with_motif, seqs_removed_motifs)

def get_avg_preds(seqs_removed, model):
    seqs_removed = np.array(seqs_removed)
    N,B,L,C = seqs_removed.shape
    removed_preds = embed.predict_np((seqs_removed.reshape(N*B,L,C)), model,
                                     batch_size=32, reshape_to_2D=False)#[:,:,cell_line]
    _,L,C = removed_preds.shape
#     removed_preds = removed_preds.reshape(N,B,L)
    avg_removed_preds = removed_preds.reshape(N,B,L,C).mean(axis=1)
    return avg_removed_preds

def get_delta_per_seq(onehot_seq, motif_pattern, model, cell_line, save_both=False):
    str_seq = ''.join(onehot_to_str(onehot_seq))
    grad_times_input = get_saliency_values(onehot_seq, model, cell_line)
    motifs_and_indices = find_multiple_motifs([motif_pattern], str_seq,
                         saliency_values=grad_times_input, filter_window=256)
    selected_seq = onehot_seq.copy()
    modified_seqs = randomize_motif_in_seq(motifs_and_indices, selected_seq)
    no_motif_preds = model(modified_seqs)[:,:,cell_line]
    if save_both:
        return (predictions[i,:,cell_line], no_motif_preds).numpy().mean(axis=0).sum()
    else:
        return (predictions[i,:,cell_line] - no_motif_preds).numpy().mean(axis=0).sum()
