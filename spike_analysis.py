import mat73
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import random
from matplotlib.colors import BoundaryNorm
from scipy.io import loadmat
from scipy.sparse import lil_matrix, csr_matrix
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import curve_fit
from itertools import combinations
import seaborn as sns
import operator

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

DIR = 'Dataset1' # DatasetRaleigh
FILE_NAME = 'eventTimes.mat' # APCdata.mat, eventTimes.mat
DATA = f'{DIR}/{FILE_NAME}'
NEURON_NUMBER = 3
RECORDING_LENGTH = 25200 # 7h

N_TRIALS = 200
N_ODORS = 8
TRIALS_PER_ODOR = int(N_TRIALS / N_ODORS)

N_ZSCORES = 504
N_BINS = 120

START = -2
ODOR_START = 0
ODOR_END = 4
SPON_START = -5
SPON_END = -1
END = 10  

NUM_PRE_POST = 30
DURATION_PRE_POST = 4
INTERVAL_PRE_POST = 60

BIN_WIDTH = (END - START) / N_BINS
LAST_BIN_BEFORE_ODOR = int(-START / BIN_WIDTH)
BINS = np.arange(START, END, BIN_WIDTH)
TRIALS = np.arange(N_TRIALS)
ODOR_PRESENTATION_SPACE = np.arange(0, TRIALS_PER_ODOR)

COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f'   # gray
]

CMAP_CLUSTER = 'tab20'

def gen_M(neurons, dt):
    ### M generation takes long time, even with dt=0.1 s...
    ### Recording time is 7h. How did Sam do it?
    T = int(RECORDING_LENGTH / dt)
    M = lil_matrix((N_NEURONS, T))
    bins = np.linspace(0, T, T+1)

    for i in range(N_NEURONS):
        neuron = np.array(neurons[i], dtype=float).flatten()
        indices = np.searchsorted(neuron, bins)
        M[i, :] = np.diff(indices)
    
    M = M.tocsr().todense()
    return M

def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c


# limit pairwise conversions 
# ASK SAM how to compute auto correlogram, Andrew seems to remember he has some clever way of doing it
def gen_fig_a(neuron, tmin=-20, tmax=20, bin_width=0.1):
    # Computes pairwise intervals between stims

    n_spikes = len(neuron)

    # Create bins
    edges = np.arange(tmin, tmax + bin_width, bin_width)
    n_bins = len(edges) - 1
    counts = np.zeros(n_bins, dtype=int)

    # Loop over each spike
    for k in range(n_spikes):
        # distances to all other spikes
        d = neuron - neuron[k]

        # Exclude self
        d = np.delete(d, k)

        # Loop over bins
        for i in range(n_bins):
            left = edges[i]
            right = edges[i + 1]
            # Count distances within this bin
            counts[i] += np.sum((d >= left) & (d < right))

    # Bin centers
    bin_centers = edges[:-1] + bin_width / 2
    plt.figure(figsize=(8,4))
    plt.bar(bin_centers, counts, width=0.5, color='gray', edgecolor='black')
    plt.xlabel("Lag (s)")
    plt.ylabel("Counts")
    plt.title("Autocorrelogram")
    plt.xlim([tmin, tmax])
    plt.show()

def trial_odor_pairs(neuron, odor_starts, start, end, sorted=False, sorted_odor_ind=None):
    trials = []
    for t in odor_starts:
        cond = (neuron > t + start) & (neuron < t + end)
        trials.append(neuron[cond] - t)

    # sort rasters based on odor number
    if sorted:
        trials = [trials[i] for i in sorted_odor_ind]

    return trials

def neuron_spike_rates(trials):
    neuron_spike_rate = np.zeros((N_TRIALS, N_BINS-1), dtype=float)
    for i, trial in enumerate(trials):
        counts, edges = np.histogram(trial, bins=BINS)
        neuron_spike_rate[i] = counts

    groups = neuron_spike_rate.reshape(N_ODORS, TRIALS_PER_ODOR, -1) # 8 x (25, N_BINS)
    spike_rate_odor_mu = np.mean(groups, axis=1) / (TRIALS_PER_ODOR * BIN_WIDTH) # (8, N_BINS)

    return spike_rate_odor_mu, edges


def gen_fig_c(neurons, odor_starts, odors):
    neuron = np.array(neurons[NEURON_NUMBER], dtype=float).flatten()
    sorted_odor_ind = np.argsort(odors, kind='stable')
    odors = odors[sorted_odor_ind]
    trials = trial_odor_pairs(neuron, odor_starts, START, END, sorted=True, sorted_odor_ind=sorted_odor_ind)

    plt.style.use('seaborn-v0_8-talk')
    print()
    _, axs = plt.subplots(1, 2, figsize=(12, 8))
    for i, spikes in enumerate(trials):
        color = COLORS[odors[i] - 1]
        
        axs[0].eventplot(spikes,
                    orientation='horizontal',
                    colors=color,
                    lineoffsets=i + 1,
                    linelengths=0.9,
                    linewidths=1.0)
        
    axs[0].set_xlim(START, END)
    axs[0].set_ylim(0.5,  + 0.5)
    tick_positions = np.arange(1, N_TRIALS + 1, TRIALS_PER_ODOR)   # 1, 26, 51, ..., 176
    axs[0].set_yticks(tick_positions)
    axs[0].set_yticklabels([f'{k}' for k in tick_positions], fontsize=9)

    axs[0].axvline(x=ODOR_START, color='k', linestyle='--', label='Odor Start')
    axs[0].axvline(x=ODOR_END, color='red', linestyle='--', label='Odor End')
    axs[0].axvspan(ODOR_START, ODOR_END, color='yellow', alpha=0.2)

    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Trials Sorted by Odor Presentation')
    axs[0].set_title(f'Spike Raster: Neuron {NEURON_NUMBER}')
    axs[0].grid(True, axis='x', alpha=0.25, linestyle=':')

    spike_rate_odor_mu, edges = neuron_spike_rates(trials)
    for i in range(N_ODORS):
        axs[1].plot(edges[:-1], spike_rate_odor_mu[i])
    axs[1].axvline(x=ODOR_START, color='k', linestyle='--', label='Odor Start')
    axs[1].axvline(x=ODOR_END, color='red', linestyle='--', label='Odor End')
    axs[1].axvspan(ODOR_START, ODOR_END, color='yellow', alpha=0.2)
    axs[1].set_xlabel('Time from odor onset (s)')
    axs[1].set_ylabel('Firing Rate (Hz)')
    axs[1].set_title(f'Spike Rate: Neuron {NEURON_NUMBER}')
    plt.show()

def gen_fig_1f(neurons, odor_starts, odors):
    z_scores = np.zeros((N_ZSCORES, N_BINS-1), dtype=float)

    for i in range(N_ZSCORES // N_ODORS):
        neuron = np.array(neurons[i+200], dtype=float).flatten()
        sorted_odor_ind = np.argsort(odors, kind='stable')
        odors = odors[sorted_odor_ind]
        trials = trial_odor_pairs(neuron, odor_starts, START, END, sorted=True, sorted_odor_ind=sorted_odor_ind)
        spike_rate_odor_mu, _ = neuron_spike_rates(trials)

        spikes_without_odor = spike_rate_odor_mu[:, :LAST_BIN_BEFORE_ODOR]
        spike_rate_mu = np.mean(spikes_without_odor)
        spike_rate_std = np.std(spikes_without_odor)

        for j in range(N_ODORS):
            idx = i*N_ODORS + j
            z_scores[idx] = (spike_rate_odor_mu[j] - spike_rate_mu) / spike_rate_std # (1, N_BINS)
    
    z_scores = z_scores[(-np.nansum(z_scores, axis=1)).argsort()]

    times = np.array([-2, 0, 4, 10])
    tick_positions = (times - START) / (END - START) * (N_BINS - 1)

    plt.figure(figsize=(5, 8))
    plt.style.use('seaborn-v0_8-talk')
    ax = sns.heatmap(z_scores, cmap='RdBu_r', center=0, 
                vmin=-3, vmax=3,
                cbar_kws={'label': 'Z-score'})
    plt.axvline(x=tick_positions[1], color='k', linestyle='--', label='Odor Start')
    plt.axvline(x=tick_positions[2], color='red', linestyle='--', label='Odor End')
    plt.title('Z-score Heatmap')

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(times, rotation=0)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel('Neuron Odor Pairs')
    plt.tight_layout()
    plt.show()

def gen_spike_counts(neurons, odor_starts, odors, sorted=True, return_odors=False, times=[]):
    """
    Population vectors across trials N_NEURONS x N_TRIALS
    If sorted: first 25 entries correspond to increasing presentations of odor 1, etc.
    If times = []: total spike count during 4 seconds of odor presentation
    If times = [t_start, t_end]: total spike count between t_start, t_end
    """

    t_start = times[0] if times else ODOR_START
    t_end = times[1] if times else ODOR_END
    spike_counts = np.zeros((N_NEURONS, N_TRIALS))

    sorted_odor_ind = np.argsort(odors, kind='stable')
    odors = odors[sorted_odor_ind]

    for i in range(N_NEURONS):
        neuron = np.array(neurons[i], dtype=float).flatten()
        trials = trial_odor_pairs(neuron, odor_starts, t_start, t_end, sorted=sorted, sorted_odor_ind=sorted_odor_ind)
        for j in range(N_TRIALS):
            spike_counts[i,j] = sum(1 for t in trials[j] if t_start < t < t_end) 
    
    return (spike_counts, odors) if return_odors else spike_counts

def gen_time_pairs_outside_trials(t_start, num_pairs, duration, interval):
    """
    Generate num pairs of times (t1, t2) (t3, t4) ... such that
    1. duration = t2 - t1
    2. interval = t1 - t3
    3. t1 = odor_start - interval
    4. reverse so lowest time pair is first
    """
    pairs = []
    t1 = t_start - interval
    for i in range(num_pairs):
        start = t1 - i * interval
        end = start + duration
        if start < 0:
            break
        pairs.append((start, end))
    return sorted(pairs, key=lambda x: x[0])

def gen_spike_counts_outside_trials(neurons, t_start, num_pairs, duration, interval):
    """
    pre-odor: t_start = odor_starts[0], interval > 0
    post-odor: t_start = odor_starts[-1], interval < 0
    """
    spike_counts = np.zeros((N_NEURONS, num_pairs))
    time_pairs = gen_time_pairs_outside_trials(t_start, num_pairs, duration, interval)

    for i in range(N_NEURONS):
        neuron = np.array(neurons[i], dtype=float).flatten()
        for j, (t1, t2) in enumerate(time_pairs):
            cond = (neuron > t1) & (neuron < t2)
            spike_counts[i, j] = np.sum(neuron[cond])

    return spike_counts
        

def population_vector_corrs(neurons, odor_starts, odors):
    """
    Generate L1/L2 corrs, Pearson's corr
    Average Pearson's corr between trials across all odors
    """
    spike_counts = gen_spike_counts(neurons, odor_starts, odors, sorted=False)
    spike_counts_sorted = gen_spike_counts(neurons, odor_starts, odors, sorted=True)
    corrs_by_odor = np.zeros((N_ODORS, TRIALS_PER_ODOR-1))

    # spike_counts_norm is cosine distance dot product
    spike_counts_L1_norm = spike_counts / spike_counts.sum(axis=0)
    spike_counts_L2_norm = spike_counts / np.linalg.norm(spike_counts, axis=0, keepdims=True)
    assert np.all(np.isfinite(spike_counts_L1_norm))
    assert np.all(np.isfinite(spike_counts_L2_norm))

    corr_L1 = spike_counts_L1_norm.T @ spike_counts_L1_norm
    corr_L2 = spike_counts_L2_norm.T @ spike_counts_L2_norm
    p_corr = np.corrcoef(spike_counts, rowvar=False)
    p_corr_sorted = np.corrcoef(spike_counts_sorted, rowvar=False)

    p_corr_avgs = corr_novel_familiar(neurons, odor_starts, odors, p_corr_sorted)
    
    # for i in range(N_ODORS):
    #     for j in range(TRIALS_PER_ODOR - 1):
    #         idx = i*TRIALS_PER_ODOR + j
    #         corrs_by_odor[i, j] = spike_counts_norm[:, idx] @ spike_counts_norm[:, idx+1]
    #         assert np.isclose(corrs_by_odor[i, j], corr[idx+1, idx], rtol=0.01)
    
    corrs_by_odor_new = np.zeros(N_TRIALS-1)
    for i in range(N_TRIALS-1):
        corrs_by_odor_new[i] = spike_counts_L2_norm[:, i] @ spike_counts_L2_norm[:, i+1]

    return corr_L1, corr_L2, p_corr, p_corr_sorted, p_corr_avgs, corrs_by_odor_new

def corr_novel_familiar(neurons, odor_starts, odors, p_corr_sorted):
    """
    Pearson's correlation between consecutive trials
    Averaged across all 8 odors
    """
    p_corr_avgs = np.zeros(TRIALS_PER_ODOR-1)
    spike_counts = gen_spike_counts(neurons, odor_starts, odors, sorted=True)

    for i in range(TRIALS_PER_ODOR-1):
        p_corr_sum_across_odor = 0
        for j in range(N_ODORS):
            idx = j*TRIALS_PER_ODOR + i
            p_corr_sum_across_odor += np.corrcoef(spike_counts[:, idx], spike_counts[:, idx+1])[0,1]
            assert np.isclose(p_corr_sorted[idx, idx+1], np.corrcoef(spike_counts[:, idx], spike_counts[:, idx+1])[0,1], rtol=0.01)
        p_corr_avgs[i] = p_corr_sum_across_odor / N_ODORS
    
    return p_corr_avgs

def population_pcorr(neurons, odor_starts, odors, times):
    spike_counts = gen_spike_counts(neurons, odor_starts, odors, sorted=False, times=times)
    p_corr = np.corrcoef(spike_counts, rowvar=False)

    return p_corr

def pcorr_pre_post_plotter(neurons, odor_starts, odors):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Pre, During, Post Odor PCorr')

    spike_counts_evoked04 = gen_spike_counts(neurons, odor_starts, odors, sorted=False, times=[ODOR_START, ODOR_END])
    spike_counts_evoked48 = gen_spike_counts(neurons, odor_starts, odors, sorted=False, times=[ODOR_END, ODOR_END+4])
    spike_counts_evoked08 = gen_spike_counts(neurons, odor_starts, odors, sorted=False, times=[ODOR_START, ODOR_END+4])
    spike_counts_spon = gen_spike_counts(neurons, odor_starts, odors, sorted=False, times=[SPON_START, SPON_END])

    spike_counts_pre_odor = gen_spike_counts_outside_trials(neurons, odor_starts[0], NUM_PRE_POST, DURATION_PRE_POST, INTERVAL_PRE_POST)
    spike_counts_post_odor = gen_spike_counts_outside_trials(neurons, odor_starts[-1], NUM_PRE_POST, DURATION_PRE_POST, -INTERVAL_PRE_POST)

    pre_mid = NUM_PRE_POST / 2
    odor_mid = NUM_PRE_POST + N_TRIALS / 2
    post_mid = N_TRIALS + NUM_PRE_POST + NUM_PRE_POST / 2

    spike_list = [spike_counts_evoked04, spike_counts_evoked48, spike_counts_evoked08, spike_counts_spon]
    titles = ['Odor Evoked: 0-4s', 'Odor Evoked: 4-8s', 'Odor Evoked: 0-8s', 'Spontaneous: -5 to -1s']

    for i, spike_counts in enumerate(spike_list):
        # 1007 x 260
        spikes_pre_post = np.concatenate([spike_counts_pre_odor, spike_counts, spike_counts_post_odor], axis=1)
        p_corr = np.corrcoef(spikes_pre_post, rowvar=False)

        axs[i].imshow(p_corr, vmin=0.75, vmax=1)
        axs[i].set_xticks([])
        axs[i].set_yticks([0, 30, 230])
        axs[i].set_title(titles[i])

        axs[i].axhline(NUM_PRE_POST, color='r', lw=1)
        axs[i].axvline(NUM_PRE_POST, color='r', lw=1)
        axs[i].axhline(N_TRIALS + NUM_PRE_POST, color='r', lw=1)
        axs[i].axvline(N_TRIALS + NUM_PRE_POST, color='r', lw=1)


        # x-axis labels (below the plot)
        for x, label in zip([pre_mid, odor_mid, post_mid], ['Pre-odor', 'Odor', 'Post-odor']):
            axs[i].text(x, -0.02, label, ha='center', va='top',
                    transform=axs[i].get_xaxis_transform(), fontsize=9, fontweight='bold')

    plt.show()

    

def cluster_corr_matrix(p_corr, k, method='average'):
    """
    corr_matrix: (n_trials, n_trials) correlation matrix
    odor_labels: (n_trials,) array/list of odor identity per trial
    k: number of clusters
    """

    # clean up rounding errors
    dist = 1 - p_corr
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)
    cluster_ids = fcluster(Z, t=k, criterion='maxclust')

    order = np.argsort(cluster_ids)
    sorted_cluster_corr = p_corr[order][:, order]
    sorted_clusters = cluster_ids[order]

    return cluster_ids, sorted_cluster_corr, sorted_clusters

def corr_matrix_plot_template(ax, k, cmap, sorted_cluster_corr, sorted_clusters, title):
    boundaries = np.where(np.diff(sorted_clusters) != 0)[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color='k', lw=1)
        ax.axvline(b, color='k', lw=1)
    unique_clusters = np.unique(sorted_clusters)
    tick_positions = []
    tick_labels = []
    for c in unique_clusters:
        idx = np.where(sorted_clusters == c)[0]
        mid = (idx[0] + idx[-1]) / 2
        tick_positions.append(mid)
        tick_labels.append(str(c))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_title(title)
    ax.set_xlabel('Cluster ID')

    n = sorted_cluster_corr.shape[0]
    dot_x = -0.03 * n  # position just left of the matrix, scaled to matrix size
    for c, y in zip(unique_clusters, tick_positions):
        color = plt.get_cmap(cmap)(c/k)
        ax.plot(dot_x, y, marker='o', markersize=6,
                color=color, clip_on=False,
                markeredgecolor='k', markeredgewidth=0.4)
    ax.set_xlim(dot_x - 0.03 * n, n - 0.5)  
    

def cluster_pcorr_plotter(p_corr, odors, ax1, ax2, k):
    CLUSTER5_MAPPING = {5: 4, 8: 5, 12: 7}

    cluster_ids, sorted_cluster_corr, sorted_clusters = cluster_corr_matrix(p_corr, k)
    ax1.imshow(sorted_cluster_corr, vmin=0.75, vmax=1)
    boundaries = np.where(np.diff(sorted_clusters) != 0)[0] + 0.5
    for b in boundaries:
        ax1.axhline(b, color='k', lw=1)
        ax1.axvline(b, color='k', lw=1)
    unique_clusters = np.unique(sorted_clusters)
    tick_positions = []
    tick_labels = []
    for c in unique_clusters:
        idx = np.where(sorted_clusters == c)[0]
        mid = (idx[0] + idx[-1]) / 2
        tick_positions.append(mid)
        tick_labels.append(str(c))
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax1.set_yticks(tick_positions)
    ax1.set_yticklabels(tick_labels)
    ax1.set_title(f'Clustered PCorr Matrix (k={k})')
    ax1.set_xlabel('Cluster ID')

    cmap = plt.get_cmap(CMAP_CLUSTER, k)
    norm = BoundaryNorm(np.arange(0.5, k + 1.5), cmap.N)
    n = sorted_cluster_corr.shape[0]
    dot_x = -0.03 * n  # position just left of the matrix, scaled to matrix size
    for c, y in zip(unique_clusters, tick_positions):
        ax1.plot(dot_x, y, marker='o', markersize=6,
                color=cmap(norm(c)), clip_on=False,
                markeredgecolor='k', markeredgewidth=0.4)
    ax1.set_xlim(dot_x - 0.03 * n, n - 0.5)  

    special_clusters = [1, 2, CLUSTER5_MAPPING[k]]
    special_borders = ['red', 'black', 'blue']
    conditions = [cluster_ids == c for c in special_clusters]
    edge_colors = np.select(conditions, special_borders, default='k')
    edge_widths = np.where(np.isin(cluster_ids, [1, 2, CLUSTER5_MAPPING[k]]), 1.6, 0.3)
    sc = ax2.scatter(odors, np.arange(len(odors)), c=cluster_ids, cmap=CMAP_CLUSTER,
                    s=20, edgecolor=edge_colors, linewidth=edge_widths)
    ax2.set_title('Clusters by Odor/Trial')
    ax2.set_xlabel('Odors')
    ax2.set_ylabel('Trials')

    return cluster_ids

def cluster_pcorr_vary_k(neurons, odor_starts, odors):
    p_corr = population_pcorr(neurons, odor_starts, odors, times=[SPON_START, SPON_END])
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Spontaneous (-5 to -1 secs)', fontsize=16)

    cluster_pcorr_plotter(p_corr, odors, axs[0,0], axs[1,0], k=5)
    cluster_pcorr_plotter(p_corr, odors, axs[0,1], axs[1,1], k=8)
    cluster_pcorr_plotter(p_corr, odors, axs[0,2], axs[1,2], k=12)

    plt.show()

def cluster_pcorr_plots(neurons, odor_starts, odors):
    _, _, p_corr, p_corr_sorted, _, _ = population_vector_corrs(neurons, odor_starts, odors)
    spike_counts_spon = gen_spike_counts(neurons, odor_starts, odors, sorted=False, times=[SPON_START, SPON_END]).T
    spike_counts_unsorted = gen_spike_counts(neurons, odor_starts, odors, sorted=False, times=[ODOR_START,ODOR_END]).T


    fig, axs = plt.subplots(2, 3, figsize=(20, 10))

    corr_img0 = axs[0,0].imshow(p_corr_sorted, vmin=0.75, vmax=1)
    fig.colorbar(corr_img0, ax=axs[0,0])
    odor_labels = [f'O{i}' for i in range(1, 9)]
    block_size = 25
    tick_positions = np.arange(block_size/2, 200, block_size)
    axs[0,0].set_xticks(tick_positions)
    axs[0,0].set_yticks(tick_positions)
    axs[0,0].set_xticklabels(odor_labels)
    axs[0,0].set_yticklabels(odor_labels)
    axs[0,0].set_title("PCorr by Odor")

    cluster_ids = cluster_pcorr_plotter(p_corr, odors, axs[0,1], axs[1,1], k=8)

    pca_spont = PCA(n_components=2)
    pca_spont.fit(spike_counts_spon)
    X_odor_pca_unsorted = pca_spont.transform(spike_counts_unsorted)
    sc = axs[1,0].scatter(X_odor_pca_unsorted[:, 0], X_odor_pca_unsorted[:, 1], 
                          c=cluster_ids, cmap='tab10', s=20, edgecolor='k', linewidth=0.3)
    handles, _ = sc.legend_elements()
    axs[1,0].legend(handles, [f'Cluster {c}' for c in np.unique(cluster_ids)],
                    title='Cluster', loc='best', fontsize=8)
    axs[1,0].set_xlabel('Spontaneous PC1')
    axs[1,0].set_ylabel('Spontaneous PC2')
    axs[1,0].set_title('Clusters onto Spontaneous PCs')

    plt.show()

def plot_corr_pop_vector(neurons, odor_starts, odors):
    corr_L1, corr_L2, p_corr, p_corr_sorted, p_corr_avgs, corrs_by_odor = population_vector_corrs(neurons, odor_starts, odors)

    # corr_L1 = corr_L1[:25, :25]
    # corr_L2 = corr_L2[:25, :25]
    # p_corr = p_corr[:25, :25]
    # o2 = p_corr_sorted[:]

    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    plt.subplots_adjust(wspace=0.3)
    fig.suptitle(f"{DIR} Sorted by Trial Number")

    # corr_img = axs[0].imshow(corr_L1)
    # fig.colorbar(corr_img, ax=axs[0])
    # axs[0].set_title("L1 Corr")

    corr_img1 = axs[1].imshow(p_corr_sorted, vmin=0.75, vmax=1)
    # fig.colorbar(corr_img1, ax=axs[1])
    odor_labels = [f'O{i}' for i in range(1, 9)]
    block_size = 25
    tick_positions = np.arange(block_size/2, 200, block_size)
    axs[1].set_xticks(tick_positions)
    axs[1].set_yticks(tick_positions)
    axs[1].set_xticklabels(odor_labels)
    axs[1].set_yticklabels(odor_labels)
    axs[1].set_title("Pear Corr by Odor")

    corr_img2 = axs[2].imshow(p_corr, vmin=0.75, vmax=1)
    # fig.colorbar(corr_img2, ax=axs[2])
    axs[2].set_title("Pearson's Corr by Trial")

    # O2 (x), O3 (y)
    block_O2O3 = p_corr_sorted[50:75, 25:50]
    axs[3].imshow(block_O2O3, vmin=0.75, vmax=1)
    axs[3].set_xlabel("O2")
    axs[3].set_ylabel("O3")
    axs[3].set_xticks([])
    axs[3].set_yticks([])

    # O3 (x), O3 (y)
    block_O3O3 = p_corr_sorted[50:75, 50:75]
    axs[4].imshow(block_O3O3, vmin=0.75, vmax=1)
    axs[4].set_xlabel("O3")
    axs[4].set_ylabel("O3")
    axs[4].set_xticks([])
    axs[4].set_yticks([])

    # corr_img3 = axs[3].imshow(p_corr_shuffle)
    # fig.colorbar(corr_img3, ax=axs[3])
    # axs[3].set_title("Pearson's Corr Random Shuffle")

    # axs[4].plot(p_corr_avgs)
    # axs[4].set_xlabel("Trial Pair Corrs (1 --> pcorr(x1, x2))")
    # axs[4].set_title("Pearson's Corr Between Consecutive Trials")

    # axs[2].plot(corrs_by_odor.T)
    # axs[2].plot(corrs_by_odor)
    # # axs[2].legend([f"{i}" for i in range(N_ODORS)])
    # axs[2].set_xlabel("Trials")
    # axs[2].set_xlabel("Consecutive trials corr")

    plt.show()


def trial_zscores(neurons, odor_starts, odors):
    """
    Calculate z scores across 200 trials
    ---> mu/sigma: spontaneous firing rate averaged/std across all neurons during this trial
    ---> 200 heatmaps with N_NEURONS rows each

    total_spike_times[n][t] = spike times for neuron n on trial t
    """
    z_scores = np.zeros((N_NEURONS, N_BINS-1, N_TRIALS), dtype=float)
    # mu, sigma are calculated over 2s of spontaneous firing
    # in poisson process, mu and variance scale linearly with time
    poisson_scaler = BIN_WIDTH/2

    spike_counts_spontaneous = gen_spike_counts(neurons, odor_starts, odors, sorted=False, times=[START,ODOR_START])

    for i in range(N_NEURONS):
        neuron = np.array(neurons[i], dtype=float).flatten()
        trials = trial_odor_pairs(neuron, odor_starts, START, END, sorted=False)
        for j, trial in enumerate(trials):
            trial_pop_vector_spon = spike_counts_spontaneous[:, j]
            mu_i, sigma_i = np.mean(trial_pop_vector_spon), np.std(trial_pop_vector_spon)
            counts, _ = np.histogram(trial, bins=BINS)
            # print(f'TRIAL: {trial}')
            # print(f'COUNTS: {counts} --> MU {mu_i*poisson_scaler}, {sigma_i*np.sqrt(poisson_scaler)}')
            z_scores[i, :, j] = (counts - mu_i*poisson_scaler)/(sigma_i*np.sqrt(poisson_scaler))
    
    # sort z scores based on trial 1
    sorted_neurons = np.argsort(-np.nansum(z_scores[:, :, 0], axis=1))
    # print(sorted_neurons[:50])
    z_scores = z_scores[sorted_neurons, :, :]

    _, axs = plt.subplots(1, 5, figsize=(20, 7))
    axs = axs.flatten()
    times = np.array([-2, 0, 4, 10])
    tick_positions = (times - START) / (END - START) * (N_BINS - 1)

    ODOR = 3
    trial_idxs = np.where(odors == ODOR)[0] 
    selected_trials = np.linspace(0, len(trial_idxs)-1, 5, dtype=int) 

    for i, trial in enumerate(trial_idxs[selected_trials]):
        sns.heatmap(z_scores[:, :, trial],ax=axs[i],cmap='RdBu_r',
        center=0,vmin=-2,vmax=2, cbar=False)

        axs[i].set_title(f'Odor {ODOR}, Presentation {selected_trials[i]+1}\nTrial {trial+1}')
        axs[i].set_xticks(tick_positions)
        axs[i].set_xticklabels(times, rotation=0)
        axs[i].axvline(x=tick_positions[1], color='k', linestyle='--', label='Odor Start')
        axs[i].axvline(x=tick_positions[2], color='red', linestyle='--', label='Odor End')
        # axs[i].axvspan(ODOR_START, ODOR_END, color='yellow', alpha=0.2)
        axs[i].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    # plt.savefig("zscores_by_trial.png", dpi=300, bbox_inches="tight")


def delta(neurons, odor_starts, odors, op, consecutive=False):
    # compute p value separately per odor
    # sliding window-- compare across datasets, is there a sweetspot of trials in window that are important?
    # what is effect size, what is p value, what is non vs parametric test, why require independent?
    # ----- for effect size, compare L2 norm of the groups.
    # plot the groups as histograms, look at heat maps for trials used to compute delta, and the delta itself
    # can we illuimate the reason why the groups appear to be drawn from different distributions?
    # look at correlation matrix of delta values -> see high correlation within group1 trials but not between group1 and group2
    # understand how p value is computed
    
    # do same analysis across novel/familiar odors in L1R1 dataset

    # INSTEAD OF PDF, USE CDF
    # DO SAME FOR ALL RESPONSE VECTORS
    # can we train classifier to distinguish novel from familiar, given population vector?
    # i.e. group in "novel" trials, compare to familiar trials, keep one out and predict and record accuracy (so 200 total classifiers)
    # prove there is something about novelty that is distinct from the rest
    # can p values be computed per odor?
    # find nice units for rasters in my own dataset
    """
    Generate delta_ij, the population vector difference between pairs of within-odor trials
    Determine if odor presentations {1...n_novel} are drawn from different distribution than {n_novel+1...25}
    """
    
    def sliding_window_mask(len, grp1, grp2, i, j):
        if j <= n_novel and i >= n_novel-len:
            grp1.extend(delta_ij[k, i, j, :])
        else: 
            grp2.extend(delta_ij[k, i, j, :])

    def sliding_window_james(n_novel, d, grp_novel, grp_fam, ij_pairs, k, group1_rand, group2_rand, n_base=2):
        ij_pairs_james = [(i, j) for (i, j) in ij_pairs if (j - i) <= d and i > n_base]
        for (i,j) in ij_pairs_james:
            i_rand, j_rand = random.choice(ij_pairs_james)
            if i >= n_base and j <= n_novel:
                grp_novel.extend(delta_ij[k, i, j, :])
                group1_rand.extend(delta_ij[k, i_rand, j_rand, :])
            elif i >= n_novel:
                grp_fam.extend(delta_ij[k, i, j, :])
                group2_rand.extend(delta_ij[k, i_rand, j_rand, :])
        # print(f'n_novel: {n_novel}, novel: {len(grp_novel)/N_NEURONS}, familiar: {len(grp_fam)/N_NEURONS}, rand 1: {len(group1_rand)/N_NEURONS}, rand 2: {len(group2_rand)/N_NEURONS}')

    delta_ij = np.zeros((N_ODORS, TRIALS_PER_ODOR, TRIALS_PER_ODOR, N_NEURONS))
    odor_idxs = np.zeros((N_ODORS, TRIALS_PER_ODOR), dtype=int)
    spike_counts = gen_spike_counts(neurons, odor_starts, odors, sorted=False, times=[ODOR_START,ODOR_END])
    p_vals = np.zeros(TRIALS_PER_ODOR-1)
    p_vals_rand = np.zeros(TRIALS_PER_ODOR-1)
    p_vals_sliding = np.zeros((TRIALS_PER_ODOR-1, 5))

    delta_mags = np.zeros((TRIALS_PER_ODOR-1, 5))

    for k in range(N_ODORS):
        odor_idxs[k] = np.where(odors == k+1)[0]
        ij_pairs_ut = [(int(i), int(j)) for i, j in combinations(odor_idxs[k], 2)]
        for i,j in ij_pairs_ut:
            presentation_i = np.where(odor_idxs[k] == i)[0][0]
            presentation_j = np.where(odor_idxs[k] == j)[0][0]
            delta_ij[k, presentation_i, presentation_j, :] = op(spike_counts[:, i], spike_counts[:, j])

    ij_pairs = [(int(i), int(j)) for i, j in combinations(ODOR_PRESENTATION_SPACE, 2)]
    _, axs = plt.subplots(2, 5, figsize=(20, 10))
    plt.subplots_adjust(wspace=0.35)
    symbol = '*' if op == operator.mul else 'Δ'

    
    for n_novel in range(1, TRIALS_PER_ODOR-1):
        group_novel, group_familiar = [], []
        group_novel_sliding_2, group_familiar_sliding_2 = [], []
        group_novel_sliding_3, group_familiar_sliding_3 = [], []
        group_novel_sliding_4, group_familiar_sliding_4 = [], []
        group1_rand, group2_rand = [], []
        group_novel_james_4, group_familiar_james_4 = [], []
        group_novel_james_6, group_familiar_james_6 = [], []
        for k in range(N_ODORS):
            if consecutive: 
                ij_pairs = np.stack([ODOR_PRESENTATION_SPACE[:-1], ODOR_PRESENTATION_SPACE[1:]], axis=1)
            for i,j in ij_pairs:
                i_rand, j_rand = random.choice(ij_pairs)
                g, g_rand = (group_novel, group1_rand) if j <= n_novel else (group_familiar, group2_rand)                 
                g.extend(delta_ij[k, i, j, :])
                # g_rand.extend(delta_ij[k, i_rand, j_rand, :])

                # sliding_window_mask(2, group_novel_sliding_2, group_familiar_sliding_2, i, j)
                # sliding_window_mask(3, group_novel_sliding_3, group_familiar_sliding_3, i, j)
                # sliding_window_mask(4, group_novel_sliding_4, group_familiar_sliding_4, i, j)
            
            n_base = 2
            sliding_window_james(n_novel, 4, group_novel_james_4, group_familiar_james_4, ij_pairs, k, group1_rand, group2_rand, n_base)
            sliding_window_james(n_novel, 6, group_novel_james_6, group_familiar_james_6, ij_pairs, k, group1_rand, group2_rand, n_base)

        p_vals[n_novel] = mannwhitneyu(group_novel, group_familiar, alternative='two-sided')[1]
        if n_novel > n_base + 1:
            p_vals_rand[n_novel] = mannwhitneyu(group1_rand, group2_rand, alternative='two-sided')[1]
        # p_vals_sliding[n_novel, 0] = mannwhitneyu(group_novel_sliding_2, group_familiar_sliding_2, alternative='two-sided')[1]
        # p_vals_sliding[n_novel, 1] = mannwhitneyu(group_novel_sliding_3, group_familiar_sliding_3, alternative='two-sided')[1]
        # p_vals_sliding[n_novel, 2] = mannwhitneyu(group_novel_sliding_4, group_familiar_sliding_4, alternative='two-sided')[1]
        if n_novel > n_base + 1:
            p_vals_sliding[n_novel, 3] = mannwhitneyu(group_novel_james_4, group_familiar_james_4, alternative='two-sided')[1]
            p_vals_sliding[n_novel, 4] = mannwhitneyu(group_novel_james_6, group_familiar_james_6, alternative='two-sided')[1]

        # for i, arr in enumerate([group_novel, group_novel_sliding_2, group_novel_sliding_3, group_novel_sliding_4, group1_rand]):
        for i, arr in enumerate([group_novel, group_novel_sliding_4, group_novel_james_4, group_novel_james_6, group1_rand]):
            chunks = [arr[i:i+N_NEURONS] for i in range(0, len(arr), N_NEURONS)]
            l2_norms = [np.linalg.norm(chunk) for chunk in chunks]
            average_l2 = np.mean(l2_norms)
            delta_mags[n_novel, i] = average_l2

        if n_novel == 9 or n_novel == 18:
            y_axs = 2 if n_novel == 18 else 1
            bins = 300
            range_vals = (-50, 50)

            counts_novel, bin_edges = np.histogram(group_novel_james_6, bins=bins, range=range_vals)
            counts_familiar, _ = np.histogram(group_familiar_james_6, bins=bins, range=range_vals)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            cdf_novel = np.cumsum(counts_novel.astype(float)/counts_novel.sum())
            cdf_familiar = np.cumsum(counts_familiar.astype(float)/counts_familiar.sum())
            cdf_diff = cdf_novel - cdf_familiar
            # cdf_diff_value = np.mean(group_novel_sliding_4) - np.mean(group_familiar_sliding_4)
            cdf_diff_value = np.trapz(cdf_diff, bin_centers)

            axs[0, y_axs].plot(bin_centers, cdf_novel, label=f'Novel {n_novel}, Sliding Window 6')
            axs[0, y_axs].plot(bin_centers, cdf_familiar, label=f'Familiar {n_novel}, Sliding Window 6')
            colors = ['brown', 'green']
            axs[0, 3].plot(bin_centers, cdf_diff, label=f'Diff {n_novel}, Sum = {cdf_diff_value:.2f}', color=colors[y_axs-1])

            axs[0, y_axs].legend()
            axs[0, y_axs].legend()
            axs[0, 3].legend()

        # print(f'{n_novel}: novel: {len(group_novel)/1007}, avg: {np.mean(group_novel):.2f}, familiar: {len(group_familiar)/1007}, avg: {np.mean(group_familiar):.2f} p: {p_vals[n_novel]}')
        # print(f'{n_novel}: novel: {len(group1_rand)/1007}, avg: {np.mean(group1_rand):.2f}, familiar: {len(group2_rand)/1007}, avg: {np.mean(group2_rand):.2f} p: {p_vals_rand[n_novel]}')
        # print(f'{n_novel}: novel: {len(group_novel_sliding_4)/1007}, avg: {np.mean(group_novel_sliding_4):.2f}, familiar: {len(group_familiar_sliding_4)/1007}, avg: {np.mean(group_familiar_sliding_4):.2f} p: {p_vals_sliding[n_novel]}')
        # print(p_vals_sliding)

    n_novel_range = np.arange(0, len(p_vals))

    # axs[0, 0].plot(n_novel_range, p_vals, marker='o', label=f'{symbol} Upper Triangular')
    # axs[0, 0].plot(n_novel_range, p_vals_sliding[:, 0], marker='x', label='δ Sliding window 2')
    # axs[0, 0].plot(n_novel_range, p_vals_sliding[:, 1], marker='x', label='δ Sliding window 3')
    # axs[0, 0].plot(n_novel_range, p_vals_sliding[:, 2], marker='x', label=f'{symbol} 4 Sliding window')
    axs[0, 0].plot(n_novel_range, p_vals_sliding[:, 3], marker='x', label=f'{symbol}: Sliding Window 4')
    axs[0, 0].plot(n_novel_range, p_vals_sliding[:, 4], marker='x', label=f'{symbol}: Sliding Window 6')
    axs[0, 0].plot(n_novel_range, p_vals_rand, marker='s', label=f'Random')
    axs[0, 0].set_yscale('log')   
    # axs[0, 0].set_ylim(1e-22, 1) 
    axs[0, 0].set_xlabel('Number of novel odors')
    axs[0, 0].set_ylabel('p')
    axs[0, 0].set_title(f'Mann-Whitney: {DIR}')
    # axs[0, 0].legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
    # axs[0, 0].legend(fontsize=8)
    axs[0, 0].grid(True)

    # axs[0, 4].plot(n_novel_range, delta_mags[:, 0], marker='o', label=f'{symbol} Upper Triangular')
    # axs[0, 4].plot(n_novel_range, delta_mags[:, 1], marker='x', label='δ Sliding window 2')
    # axs[0, 4].plot(n_novel_range, delta_mags[:, 2], marker='x', label='δ Sliding window 3')
    # axs[0, 4].plot(n_novel_range, delta_mags[:, 1], marker='x', label=f'{symbol} 4 Sliding window')
    axs[0, 4].plot(n_novel_range, delta_mags[:, 2], marker='x', label=f'{symbol}=4 Sliding Window')
    axs[0, 4].plot(n_novel_range, delta_mags[:, 3], marker='x', label=f'{symbol}=6 Sliding Window')
    axs[0, 4].plot(n_novel_range, delta_mags[:, 4], marker='s', label=f'Random')
    axs[0, 4].set_xlabel('Number of novel odors')
    axs[0, 4].set_ylabel(f'|{symbol}|₂')
    axs[0, 4].legend()

    # plt.show()
    return axs


def x(neurons, odor_starts, odors, axs):
    """
    Determine if odor presentations {1...n_novel} are drawn from different distribution than {n_novel+1...25}
    For population vectors x
    """
    
    def sliding_window_mask(len, grp1, grp2, i):
        grp = grp1 if i >= n_novel-len and i <= n_novel else grp2
        grp.extend(x[k, i, :])

    spike_counts = gen_spike_counts(neurons, odor_starts, odors, sorted=True, times=[ODOR_START,ODOR_END])
    p_vals = np.zeros(TRIALS_PER_ODOR-1)
    p_vals_rand = np.zeros(TRIALS_PER_ODOR-1)
    p_vals_sliding = np.zeros((TRIALS_PER_ODOR-1, 3))

    # spike_counts.shape = (N_NEURONS, N_TRIALS)
    x = np.zeros((N_ODORS, TRIALS_PER_ODOR, N_NEURONS))
    x = spike_counts.reshape(N_NEURONS, N_ODORS, TRIALS_PER_ODOR).transpose(1, 2, 0)
    # for i in range(N_ODORS):
    #     for j in range(TRIALS_PER_ODOR):
    #         assert x[i, j, :].all() == spike_counts[:, i*TRIALS_PER_ODOR + j].all()

    x_mags = np.zeros((TRIALS_PER_ODOR-1, 5))

    
    for n_novel in range(2, TRIALS_PER_ODOR-1):
        group_novel, group_familiar = [], []
        group_novel_sliding_2, group_familiar_sliding_2 = [], []
        group_novel_sliding_3, group_familiar_sliding_3 = [], []
        group_novel_sliding_4, group_familiar_sliding_4 = [], []
        group1_rand, group2_rand = [], []
        n_base = 2
        for k in range(N_ODORS):
            # James's sliding window for X, cut off first n_base trials and rerun analysis
            for i in range(n_base, TRIALS_PER_ODOR):
                g, g_rand = (group_novel, group1_rand) if i <= n_novel else (group_familiar, group2_rand)
                g.extend(x[k, i, :])
                g_rand.extend(x[k, np.random.choice(25), :])

                sliding_window_mask(2, group_novel_sliding_2, group_familiar_sliding_2, i)
                sliding_window_mask(3, group_novel_sliding_3, group_familiar_sliding_3, i)
                sliding_window_mask(4, group_novel_sliding_4, group_familiar_sliding_4, i)

        p_vals[n_novel] = mannwhitneyu(group_novel, group_familiar, alternative='two-sided')[1]
        p_vals_rand[n_novel] = mannwhitneyu(group1_rand, group2_rand, alternative='two-sided')[1]
        p_vals_sliding[n_novel, 0] = mannwhitneyu(group_novel_sliding_2, group_familiar_sliding_2, alternative='two-sided')[1]
        p_vals_sliding[n_novel, 1] = mannwhitneyu(group_novel_sliding_3, group_familiar_sliding_3, alternative='two-sided')[1]
        p_vals_sliding[n_novel, 2] = mannwhitneyu(group_novel_sliding_4, group_familiar_sliding_4, alternative='two-sided')[1]

        for i, arr in enumerate([group_novel, group_novel_sliding_2, group_novel_sliding_3, group_novel_sliding_4, group1_rand]):
            chunks = [arr[i:i+N_NEURONS] for i in range(0, len(arr), N_NEURONS)]
            l2_norms = [np.linalg.norm(chunk) for chunk in chunks]
            average_l2 = np.mean(l2_norms)
            x_mags[n_novel, i] = average_l2

        if n_novel == 5 or n_novel == 18:
            y_axs = 2 if n_novel == 18 else 1
            bins = 300
            range_vals = (-50, 50)

            counts_novel, bin_edges = np.histogram(group_novel, bins=bins, range=range_vals)
            counts_familiar, _ = np.histogram(group_familiar, bins=bins, range=range_vals)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            cdf_novel = np.cumsum(counts_novel.astype(float)/counts_novel.sum())
            cdf_familiar = np.cumsum(counts_familiar.astype(float)/counts_familiar.sum())
            cdf_diff = cdf_novel - cdf_familiar
            # cdf_diff_value = np.mean(group_novel_sliding_3) - np.mean(group_familiar_sliding_3)
            cdf_diff_value = np.trapz(cdf_diff, bin_centers)

            axs[1, y_axs].plot(bin_centers, cdf_novel, label=f'Novel {n_novel}, X')
            axs[1, y_axs].plot(bin_centers, cdf_familiar, label=f'Familiar {n_novel}, X')
            colors = ['brown', 'green']
            axs[1, 3].plot(bin_centers, cdf_diff, label=f'Diff {n_novel}, Sum = {cdf_diff_value:.2f}', color=colors[y_axs-1])
            
            axs[1, y_axs].legend()
            axs[1, y_axs].legend()
            axs[1, 3].legend()

        # print(f'{n_novel}: novel: {len(group_novel)/1007}, avg: {np.mean(group_novel):.2f}, familiar: {len(group_familiar)/1007}, avg: {np.mean(group_familiar):.2f} p: {p_vals[n_novel]}')
        # print(f'{n_novel}: novel: {len(group1_rand)/1007}, avg: {np.mean(group1_rand):.2f}, familiar: {len(group2_rand)/1007}, avg: {np.mean(group2_rand):.2f} p: {p_vals_rand[n_novel]}')
        # print(f'{n_novel}: novel: {len(group_novel_sliding_4)/1007}, avg: {np.mean(group_novel_sliding_4):.2f}, familiar: {len(group_familiar_sliding_4)/1007}, avg: {np.mean(group_familiar_sliding_4):.2f} p: {p_vals_sliding[n_novel]}')
        # print(p_vals_sliding)

    n_novel_range = np.arange(0, len(p_vals))

    axs[1, 0].plot(n_novel_range, p_vals, marker='o', label='X (n_base = 2)')
    axs[1, 0].plot(n_novel_range, p_vals_sliding[:, 0], marker='x', label='X Sliding window 2')
    axs[1, 0].plot(n_novel_range, p_vals_sliding[:, 1], marker='x', label='X Sliding window 3')
    axs[1, 0].plot(n_novel_range, p_vals_sliding[:, 2], marker='x', label='X Sliding window 4')
    axs[1, 0].plot(n_novel_range, p_vals_rand, marker='s', label='X Random')
    axs[1, 0].set_yscale('log')    
    axs[1, 0].set_xlabel('Number of novel odors')
    axs[1, 0].set_ylabel('p')
    axs[1, 0].set_title(f'Mann-Whitney: {DIR}')
    axs[1, 0].legend(fontsize=8)
    axs[1, 0].grid(True)

    axs[1, 4].plot(n_novel_range, x_mags[:, 0], marker='o', label='X (n_base = 2)')
    axs[1, 4].plot(n_novel_range, x_mags[:, 1], marker='x', label='X Sliding window 2')
    axs[1, 4].plot(n_novel_range, x_mags[:, 2], marker='x', label='X Sliding window 3')
    axs[1, 4].plot(n_novel_range, x_mags[:, 3], marker='x', label='X Sliding window 4')
    axs[1, 4].plot(n_novel_range, x_mags[:, 4], marker='s', label='X Random')
    axs[1, 4].set_xlabel('Number of novel odors')
    axs[1, 4].set_ylabel('|X|₂')
    axs[1, 4].legend()

    plt.show()



def svm(neurons, odor_starts, odors):
    accuracy_linear = np.zeros(TRIALS_PER_ODOR)
    
    spike_counts = gen_spike_counts(neurons, odor_starts, odors, sorted=True, times=[ODOR_START,ODOR_END]).T
    spike_counts_spontaneous, odors_spon = gen_spike_counts(neurons, odor_starts, odors, sorted=True, return_odors=True, times=[-12,-2])
    spike_counts_spontaneous = spike_counts_spontaneous.T

    loo = LeaveOneOut()
    clf_linear = SVC(kernel='linear', C=100)

    predictions_linear = np.zeros(N_TRIALS, dtype=np.int64)
    truth = np.zeros(N_TRIALS, dtype=np.int64)

    for n_novel in range(1, TRIALS_PER_ODOR-1):
        #  NOVEL, FAMILIAR = 0, 1
        #  [ 0 .. n_novel .. 0 1 ... 1 | ... ]
        Y = np.tile(np.r_[np.zeros(n_novel), np.ones(25-n_novel)], TRIALS_PER_ODOR)[:N_TRIALS].astype(int)

        for i, (train_idx, test_idx) in enumerate(loo.split(spike_counts)):
        
            X_train, X_test = spike_counts[train_idx], spike_counts[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            clf_linear.fit(X_train, Y_train)
            predictions_linear[i] = (clf_linear.predict(X_test)[0])

            truth[i] = (Y_test[0])

        accuracy_linear[n_novel] = accuracy_score(truth, predictions_linear)

        # num_novel, num_familiar = N_TRIALS - sum(Y), sum(Y)
        # num_novel_correct, num_familiar_correct = np.sum((predictions == NOVEL) & (truth == NOVEL)), np.sum((predictions == FAMILIAR) & (truth == FAMILIAR))
        # print(f"{n_novel}: Accuracy: {accuracy[n_novel]:.4f}, % novel: {num_novel/N_TRIALS}, % novel correct: {num_novel_correct/num_novel},  % familiar corect; {num_familiar_correct/num_familiar}")

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    plt.subplots_adjust(wspace=0.35)

    pca_spont = PCA(n_components=2)
    X_spontaneous_pca = pca_spont.fit_transform(spike_counts_spontaneous)

    pca_odor = PCA(n_components=2)
    X_pca = pca_odor.fit_transform(spike_counts)

    # spontaneous_variance = pca_spont.explained_variance_ratio_ * 100
    # odor_evoked_variance = pca_odor.explained_variance_ratio_ * 100

    # axs[0].plot(pca_odor.components_[0], label=f'PC1 ({odor_evoked_variance[0]:.1f}%)')
    # axs[0].plot(pca_odor.components_[1], label=f'PC2 ({odor_evoked_variance[1]:.1f}%)')

    # axs[0].plot(pca_spont.components_[0], label=f'Spon. PC1 ({spontaneous_variance[0]:.1f}%)')
    # axs[0].plot(pca_spont.components_[1], label=f'Spon. PC2 ({spontaneous_variance[1]:.1f}%)')

    # axs[0].set_xlabel("Neuron")
    # axs[0].set_ylabel("PC Loading Weight")
    # axs[0].legend()
    # axs[0].plot(ODOR_PRESENTATION_SPACE, accuracy_linear*100, label='Linear SVM')
    # axs[0].axhline(50, color='red', linestyle='--', label='Baseline')
    # axs[0].set_xlabel('# Novel Presentations')
    # axs[0].set_ylabel('Accuracy')
    # axs[0].set_xlim(1, TRIALS_PER_ODOR-2)
    # axs[0].set_ylim(40, 100)
    # axs[0].set_title(f'{DIR}: Spontaneous Activity Classification Accuracy')
    # axs[0].legend()

    # SVM visualization
    n_novel_axis = np.arange(2, 25, 1)
    svm_movement = []
    axis = -1

    # X_odor_pca = pca_spont.transform(spike_counts)
    X_odor_pca = pca_spont.transform(spike_counts_spontaneous)

    padding_x, padding_y = 70, 0.5
    x_min, x_max = X_odor_pca[:, 0].min() - padding_x, X_odor_pca[:, 0].max() + padding_x
    y_min, y_max = X_odor_pca[:, 1].min() - padding_y, X_odor_pca[:, 1].max() + padding_y
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                        np.linspace(y_min, y_max, 500))
    
    for i, n_novel in enumerate(n_novel_axis):
        Y = np.tile(np.r_[np.zeros(n_novel), np.ones(25-n_novel)], TRIALS_PER_ODOR)[:N_TRIALS].astype(int)

        # pca = PCA(n_components=2)
        # X_spontaneous_pca = pca.fit_transform(spike_counts_spontaneous)
        # X_2d = StandardScaler().fit_transform(X_spontaneous_pca)

        # fit odor evoked activity to spontaneous PCs

        
        clf_pca_linear = SVC(kernel='linear', C=0.1)
        clf_pca_linear.fit(X_odor_pca, Y)
        Z_linear = clf_pca_linear.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z_linear = Z_linear.reshape(xx.shape)

        w = clf_pca_linear.coef_[0]
        b = clf_pca_linear.intercept_[0]
        center = -b * w / np.dot(w, w)
        svm_movement.append(center)


        odor_colors = np.array(['#e6194b', '#3cb44b', '#4363d8', '#f58231',
                         '#911eb4', '#42d4f4', '#f032e6', '#bfef45'])
        if n_novel in [3, 10, 17]: 
            point_colors = odor_colors[odors_spon - 1]
            scatter = axs[axis+1].scatter(X_odor_pca[:, 0], X_odor_pca[:, 1], c=point_colors, edgecolor='k', s=40)

            # Build the legend manually since we're not using a colormap
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markeredgecolor='k', markersize=8) for c in odor_colors]
            axs[axis+1].legend(handles, [f'Odor {i}' for i in range(1, 9)], loc="upper right", ncol=2, fontsize=8)
            # scatter = axs[axis+1].scatter(X_odor_pca[:, 0], X_odor_pca[:, 1], c=odors_spon, cmap='coolwarm', edgecolor='k', s=40)
            # handles, _ = scatter.legend_elements()
            # axs[axis+1].legend(handles, ['Novel', 'Familiar'], loc="upper right")
            # axs[axis+1].set_title(f"2D Linear SVM, n_novel = {n_novel}\nExplained Variance: {np.sum(pca_spont.explained_variance_ratio_)*100:.1f}%")
            axs[axis+1].set_title(f"n_novel = {n_novel}\nExplained Variance: {np.sum(pca_spont.explained_variance_ratio_)*100:.1f}%")
            axs[axis+1].set_xlabel("Spontaneous PC 1")
            axs[axis+1].set_ylabel("Spontaneous PC 2")

            axis += 1

    svm_movement = np.array(svm_movement)   


    spike_counts_spontaneous_unsorted = gen_spike_counts(neurons, odor_starts, odors, sorted=False, times=[-12,-2]).T
    X_odor_pca_unsorted = pca_spont.transform(spike_counts_spontaneous_unsorted)
    colors = np.arange(X_odor_pca_unsorted.shape[0])
    scatter = axs[3].scatter(
        X_odor_pca_unsorted[:, 0],
        X_odor_pca_unsorted[:, 1],
        c=colors,
        cmap='coolwarm',
        edgecolor='k',
        s=40
    )
    cbar = fig.colorbar(scatter, ax=axs[3])
    cbar.set_label("Trial Number")
    # axs[3].scatter(X_odor_pca[:, 0], X_odor_pca[:, 1], color='grey', alpha=0.75, cmap='coolwarm', edgecolor='k', s=40)
    # axs[3].quiver(
    #     svm_movement[:-1,0],
    #     svm_movement[:-1,1],
    #     np.diff(svm_movement[:,0]),
    #     np.diff(svm_movement[:,1]),
    #     angles='xy',
    #     scale_units='xy',
    #     scale=1,         
    #     # width=0.008,
    #     # headwidth=3,
    #     # headlength=3,
    #     color='blue'
    # )
    axs[3].set_xlim(x_min, x_max)
    axs[3].set_ylim(y_min, y_max)
    axs[3].set_title("Spontaneous Population Response")
    axs[3].set_xlabel("Spontaneous PC 1")
    axs[3].set_ylabel("Spontaneous PC 2")
    
    plt.show()


    # Can we do linear regression across time of these 1007D vectors and does it fit well? Compare to null model 
    # where time is scrambled (randomly arrange 200 trials), 
    # generate distribution with 100 random attempts and see how much bigger is your r?


def odor_trial_split(neurons, odor_starts, odors):
    spike_counts = gen_spike_counts(neurons, odor_starts, odors, sorted=True, times=[ODOR_START,ODOR_END])

    X = spike_counts.T  
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    plt.style.use('seaborn-v0_8-talk')
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    colors = [
        (0.65,0.05,0.05),  # dark red
        (0.05,0.2,0.65),   # dark blue
        (0.0,0.55,0.2)     # dark green
    ]

    def lighten_color(c, amount):
        white = np.array([1,1,1])
        return tuple(c + (white - c) * amount)

    for i, odor in enumerate([1, 2, 5]):
        start = odor * 25
        end = start + 25
        pts = X_pca[start:end]

        base_color = colors[i]
        alphas = np.exp(np.linspace(0, -1.6, 25))  
        lightness = np.linspace(0, 0.7, 25)        # dark → lighter


        for t in range(25):
            color_t = lighten_color(base_color, lightness[t])     
            ax.scatter(
                pts[t,0], pts[t,1], pts[t,2],
                color=color_t,
                alpha=alphas[t],
                s=50,
                label=f'Odor {odor+1}' if t == 0 else None
            )

        for t in range(24):
            ax.plot(
                pts[t:t+2,0],
                pts[t:t+2,1],
                pts[t:t+2,2],
                linestyle=':',
                color=base_color,
                alpha=alphas[t]
            )

        z_offset = 0.02 * (X_pca[:,2].max() - X_pca[:,2].min())
        label_trials = [0,24]
        for t in label_trials:
            ax.text(
                pts[t,0],
                pts[t,1],
                pts[t,2] + z_offset,
                f'trial {t+1}',
                color=base_color,
                fontsize=10
            )

    ax.set_xlabel("PC1", labelpad=18)
    ax.set_ylabel("PC2", labelpad=18)
    ax.set_zlabel("PC3", labelpad=18)
    # ax.set_xlim(-300,300)
    # ax.set_zlim(-300,200)
    ax.legend()

    plt.title(f"{DIR}: Odor Trajectory by Presentation\nExplained Variance: {np.sum(pca.explained_variance_ratio_)*100:.1f}%")
    plt.tight_layout()
    plt.show()

def mean_pop_firing_rate(neurons, odor_starts, odors):
    # spike_counts = gen_spike_counts(neurons, odor_starts, odors, sorted=True, times=[ODOR_START,ODOR_END])
    spike_counts = gen_spike_counts(neurons, odor_starts, odors, sorted=True, times=[ODOR_START-4,ODOR_START])
    # X = spike_counts.reshape(1007, 8, 25)/(ODOR_END - ODOR_START)
    X = spike_counts.reshape(1007, 8, 25)/4

    mean = X.mean(axis=(0,1))
    std  = X.std(axis=(0,1))

    plt.errorbar(range(1,26), mean, yerr=std, fmt='o')
    plt.style.use('seaborn-v0_8-talk')
    plt.xlabel("# of Odor Presentations")
    plt.ylabel("Firing Rate (Hz)")
    plt.title(f"{DIR}: Mean Spontaneous Response Firing Rate")
    plt.show()


def pre_post_time_series(neurons, odor_starts, odors):
    BIAS = 500
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    spike_counts_unsorted = gen_spike_counts(neurons, odor_starts, odors, sorted=False, times=[ODOR_START,ODOR_END]).T
    spike_counts_spon = gen_spike_counts(neurons, odor_starts, odors, sorted=False, times=[SPON_START, SPON_END]).T

    spike_counts_pre_odor = gen_spike_counts_outside_trials(neurons, odor_starts[0], NUM_PRE_POST, DURATION_PRE_POST, INTERVAL_PRE_POST).T
    spike_counts_post_odor = gen_spike_counts_outside_trials(neurons, odor_starts[-1], NUM_PRE_POST, DURATION_PRE_POST, -INTERVAL_PRE_POST).T

    pca_spont = PCA(n_components=2)
    pca_spont.fit(spike_counts_spon)
    X_odor_pca_unsorted = pca_spont.transform(spike_counts_unsorted)
    X_pre_odor_pca = pca_spont.transform(spike_counts_pre_odor)
    X_post_odor_pca = pca_spont.transform(spike_counts_post_odor)

    X_pre_post_combined = np.vstack([X_pre_odor_pca, X_post_odor_pca])

    sc1 = axs[0].scatter(np.abs(X_odor_pca_unsorted[:, 0]), X_odor_pca_unsorted[:, 1] + BIAS, c=TRIALS, cmap='RdBu_r', s=20, edgecolor='k', linewidth=0.3)
    sc2 = axs[0].scatter(np.abs(X_pre_post_combined[:, 0]), X_pre_post_combined[:, 1] + BIAS, c=np.arange(2*NUM_PRE_POST), cmap='BuPu_r', s=20, edgecolor='k', linewidth=0.3)
    cbar2 = plt.colorbar(sc2, location='left', fraction=0.046, pad=0.2)
    cbar2.set_label('Pre-to-Post Number')
    axs[0].set_xlabel('|Spontaneous PC1|')
    axs[0].set_ylabel('Spontaneous PC2')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')

    sc1 = axs[1].scatter(X_odor_pca_unsorted[:, 0], X_odor_pca_unsorted[:, 1], c=TRIALS, cmap='RdBu_r', s=20, edgecolor='k', linewidth=0.3)
    cbar1 = plt.colorbar(sc1, location='right', fraction=0.046, pad=0.04)
    cbar1.set_label('Trial number')
    axs[1].set_xlabel('Spontaneous PC1')
    axs[1].set_ylabel('Spontaneous PC2')

    plt.show()

def single_unit_regression(neurons, odor_starts, odors):  
    # N_TRIALS x N_NEURONS
    spike_counts_unsorted = gen_spike_counts(neurons, odor_starts, odors, sorted=False, times=[ODOR_START,ODOR_END]).T
    spike_counts_spon = gen_spike_counts(neurons, odor_starts, odors, sorted=False, times=[SPON_START, SPON_END]).T

    slopes = np.zeros(N_NEURONS)
    intercepts = np.zeros(N_NEURONS)
    r_values = np.zeros(N_NEURONS)
    p_values = np.zeros(N_NEURONS)

    l2 = np.linalg.norm(spike_counts_unsorted, axis=1) 
    l2_spon = np.linalg.norm(spike_counts_spon, axis=1) 
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs[0,0].plot(TRIALS, l2, label='odor-evoked')
    axs[0,0].plot(TRIALS, l2_spon, label='spontaneous')
    axs[0,0].legend()
    axs[0,0].set_xlabel('Trial Number')
    axs[0,0].set_ylabel('L2 Population Response')
    axs[0,0].set_title('Magnitude of Population Response By Trial')

    # -------ANALYZE SPONTANEOUS--------
    spike_counts_unsorted = spike_counts_spon
    # -------ANALYZE SPONTANEOUS--------

    for i in range(N_NEURONS):
        slopes[i], intercepts[i], r_values[i], p_values[i], se = stats.linregress(TRIALS, spike_counts_unsorted[:, i])

    # Correct for multiple comparisons (Benjamini-Hochberg FDR)
    reject, _, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

    print(f"Significant neurons: {reject.sum()} / {N_NEURONS}")
    print(f"  Increasing: {(reject & (slopes > 0)).sum()}")
    print(f"  Decreasing: {(reject & (slopes < 0)).sum()}")
    
    slopes_sorted = np.argsort(slopes)
    min, max = slopes_sorted[0], slopes_sorted[-1]

    colors = np.where(reject, np.where(slopes > 0, 'crimson', 'steelblue'), 'lightgray')
    axs[0,1].bar(np.arange(N_NEURONS), slopes, color=colors)
    axs[0,1].annotate(f'{min+1}', xy=(min, slopes[min]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7, weight='bold')
    axs[0,1].annotate(f'{max+1}', xy=(max, slopes[max]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7, weight='bold')
    axs[0,1].axhline(0, color='k', linewidth=0.8)
    axs[0,1].set_xlabel('Unit')
    axs[0,1].set_ylabel('Firing rate vs. trial slope')
    axs[0,1].set_title('Single Unit Drift Slopes')

    axs[0,2].plot(TRIALS, spike_counts_unsorted[:, max], color='crimson', label=f'Unit {max+1}')
    axs[0,2].plot(TRIALS, spike_counts_unsorted[:, min], color='steelblue', label=f'Unit {min+1}')
    axs[0,2].set_xlabel('Trial number')
    axs[0,2].set_ylabel('Firing rate')
    axs[0,2].set_title('Units with greatest Δ firing rate')
    axs[0,2].legend()


    # Identify which PC correlates with trial number
    pca_spont = PCA(n_components=2)
    pca_spont.fit(spike_counts_spon)
    X_odor_pca_unsorted = pca_spont.transform(spike_counts_unsorted)

    reg = LinearRegression().fit(trial_nums.reshape(-1, 1), X_odor_pca_unsorted)
    drift_direction = reg.coef_.flatten()  # regression coefficients of odor-evoked response in odor pca
    drift_direction_unit = drift_direction / np.linalg.norm(drift_direction)

    print(f"Drift direction (unit vector in PC1/PC2 space): {drift_direction_unit}")
    print(f"  -> angle from PC1 axis: {np.degrees(np.arctan2(drift_direction_unit[1], drift_direction_unit[0])):.1f} deg")
    
    projection_onto_drift = X_odor_pca_unsorted @ drift_direction_unit  # shape (n_trials,)
    r_drift, p_drift = stats.pearsonr(trial_nums, projection_onto_drift)
    print(f"Projection vs trial number: r={r_drift:.3f}, p={p_drift:.4g}")

    sc1 = axs[1,0].scatter(X_odor_pca_unsorted[:, 0], X_odor_pca_unsorted[:, 1], c=trial_nums, cmap='RdBu_r', s=20, edgecolor='k', linewidth=0.3)
    cbar1 = plt.colorbar(sc1, label='Trial number')

    # overlay the fitted drift direction as an arrow from the data centroid
    centroid = X_odor_pca_unsorted.mean(axis=0)
    arrow_scale = np.ptp(X_odor_pca_unsorted[:, 0]) * 0.25  # scale arrow to plot size

    print(f'CENTROID: {centroid}, DRIFT DIR: {drift_direction_unit}, arrow scale: {arrow_scale}')

    axs[1,0].annotate('', xy=centroid + arrow_scale*drift_direction_unit, xytext=centroid - arrow_scale*drift_direction_unit,
                arrowprops=dict(arrowstyle='-|>', color='crimson', lw=2))
    axs[1,0].set_xlabel('Spontaneous PC1')
    axs[1,0].set_ylabel('Spontaneous PC2')
    axs[1,0].set_title(f'Spontaneous Drift Axis')


    drift_loadings_full = drift_direction_unit @ pca_spont.components_  # shape (n_neurons,)
    neg_unit, pos_unit = np.argmin(drift_loadings_full), np.argmax(drift_loadings_full)

    r_check, p_check = stats.pearsonr(drift_loadings_full, slopes)
    print(f"Composite drift loading vs. single-unit slope: r={r_check:.3f}, p={p_check:.4g}")

    colors = np.where(~reject, 'lightgray', np.where(slopes > 0, 'crimson', 'steelblue'))
    axs[1,1].scatter(drift_loadings_full, slopes, c=colors,
                edgecolor='k', s=40, alpha=0.8)
    axs[1,1].annotate(neg_unit+1, (drift_loadings_full[neg_unit], slopes[neg_unit]),
                   textcoords="offset points", xytext=(4, 4), fontsize=7, weight='bold')
    axs[1,1].annotate(pos_unit+1, (drift_loadings_full[pos_unit], slopes[pos_unit]),
                   textcoords="offset points", xytext=(4, 4), fontsize=7, weight='bold')
    axs[1,1].annotate(max+1, (drift_loadings_full[max], slopes[max]),
                   textcoords="offset points", xytext=(4, 4), fontsize=7, weight='bold')
    axs[1,1].set_xlabel('Loading Component')
    axs[1,1].set_ylabel('Firing rate vs. trial slope')
    axs[1,1].axhline(0, color='gray', lw=0.5)
    axs[1,1].axvline(0, color='gray', lw=0.5)
    axs[1,1].set_title(f'ΔFiring rate wrt loading component r={r_check:.3f}')

    axs[1,2].plot(trial_nums, spike_counts_unsorted[:, pos_unit], color='crimson', label=f'Unit {pos_unit+1}')
    axs[1,2].plot(trial_nums, spike_counts_unsorted[:, neg_unit], color='steelblue', label=f'Unit {neg_unit+1}')
    axs[1,2].set_xlabel('Trial Number')
    axs[1,2].set_ylabel('Firing Rate')
    axs[1,2].set_title(f'Top Contributing Unit Spike Counts')
    axs[1,2].legend()

    # Get that PC's neuron loadings
    # drift_loadings = pca_components[drift_pc, :]  # shape (n_neurons,)

    # # Correlate loadings with regression slopes
    # loading_slope_r, loading_slope_p = stats.pearsonr(drift_loadings, slopes)
    # print(f"Loading vs. slope correlation: r={loading_slope_r:.3f}, p={loading_slope_p:.4f}")

    # axs[3].scatter(drift_loadings, slopes, c=np.where(reject, 'crimson', 'lightgray'),
    #             edgecolor='k', s=40, alpha=0.8)
    # axs[3].set_xlabel(f'PC{drift_pc+1} loading')
    # axs[3].set_ylabel('Firing rate vs. trial slope')
    # axs[3].set_title(f'r={loading_slope_r:.2f}, p={loading_slope_p:.4f}')
    # axs[3].axhline(0, color='gray', linewidth=0.5)
    # axs[3].axvline(0, color='gray', linewidth=0.5)

    plt.show()


def run_pca(pc_set, projection_set, dimensions):
    pca = PCA(n_components=dimensions)
    pca.fit(pc_set)
    return pca.transform(projection_set), np.sum(pca.explained_variance_ratio_)

def proj_evo_spon_pca(neurons, odor_starts, odors):
    fig, axs = plt.subplots(1, 1, figsize=(8,8))
    spike_counts_evoked = gen_spike_counts(neurons, odor_starts, odors, sorted=False, times=[ODOR_START,ODOR_END]).T
    spike_counts_spon = gen_spike_counts(neurons, odor_starts, odors, sorted=False, times=[SPON_START, SPON_END]).T

    evoked_pca, var_captured = run_pca(spike_counts_spon, spike_counts_evoked, dimensions=2)
    spon_pca, _ = run_pca(spike_counts_spon, spike_counts_spon, dimensions=2)

    # make sure low trials have at least 25% opacity in their respective hues to distinguish them
    axs.scatter(evoked_pca[:, 0], evoked_pca[:, 1], c=TRIALS, cmap='Reds', edgecolor='k', s=40, vmin=-N_TRIALS/4, vmax=N_TRIALS-1, label='Evoked')
    axs.scatter(spon_pca[:, 0], spon_pca[:, 1], c=TRIALS, cmap='Blues', edgecolor='k', s=40, vmin=-N_TRIALS/4, vmax=N_TRIALS-1, label='Spontaneous')

    axs.set_xlabel('Spontaneous PC1')
    axs.set_ylabel('Spontaneous PC2')
    axs.set_title(f'Evoked & Spontaneous Firing Spaces in Spontaneous PCs ({var_captured*100:.2f}%)')
    axs.legend()

    plt.show()


def cluster_in_pca_space(neurons, odor_starts, odors, k_list, d_list):
    fig, axs = plt.subplots(3, 4, figsize=(18, 9))
    
    spike_counts_unsorted = gen_spike_counts(neurons, odor_starts, odors, sorted=False, times=[ODOR_START,ODOR_END]).T
    spike_counts_spon = gen_spike_counts(neurons, odor_starts, odors, sorted=False, times=[SPON_START, SPON_END]).T
    p_corr = population_pcorr(neurons, odor_starts, odors, times=[ODOR_START,ODOR_END])

    for i, d in enumerate(d_list):
        X_odor_pca_unsorted, variance_captured = run_pca(spike_counts_spon, spike_counts_unsorted, dimensions=d)

        for j, k in enumerate(k_list):
            Z = linkage(X_odor_pca_unsorted, method='average')
            cluster_ids = fcluster(Z, t=k, criterion='maxclust')

            # edge_colors = np.where(cluster_ids == cluster_ids[1], 'red', 'k')
            # edge_widths = np.where(cluster_ids == cluster_ids[1], 1.6, 0.3)
            axs[i,j].scatter(odors, np.arange(len(odors)), c=cluster_ids/k, cmap='RdBu_r',
                                vmin=0, vmax=1, s=20, edgecolor='k', linewidth=0.3)
            axs[i,j].set_title(f'{k} Clusters, {d} Dimensions, {variance_captured*100:.2f}%')
            axs[i,j].set_xlabel('Odors')
            axs[i,j].set_ylabel('Trials')

            if k == 8:
                order = np.argsort(cluster_ids)
                sorted_cluster_corr = p_corr[order][:, order]
                sorted_clusters = cluster_ids[order]

                img0 = axs[i,3].imshow(sorted_cluster_corr, vmin=0.75, vmax=1)
                fig.colorbar(img0, ax=axs[i,3])
                corr_matrix_plot_template(axs[i,3], k, 'RdBu_r', sorted_cluster_corr, sorted_clusters, f'PCA Clustered PCorr (d={d}, k={k})')

    plt.tight_layout()
    plt.show()

def plot_drift_across_presentations(neurons, odor_starts, odors):
    def angle_between(v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(cos_theta))

    def corr_angle_between(v1, v2):
        r = np.corrcoef(v1, v2)[0, 1]  # mean-centered pcorr
        return np.degrees(np.arccos(r))

    def drift_across_presentations(spike_counts, operation, compare_to_first=False):
        delta_pop_vector_matrix = np.zeros((N_ODORS, TRIALS_PER_ODOR-1))
        for i in range(N_ODORS):
            in_odor_group = spike_counts[i*TRIALS_PER_ODOR : (i+1)*TRIALS_PER_ODOR]
            for j in range(TRIALS_PER_ODOR-1):
                left_idx = 0 if compare_to_first else j
                delta_pop_vector_matrix[i, j] = operation(in_odor_group[left_idx], in_odor_group[j+1])

        if compare_to_first:
            delta_pop_vector_matrix = np.concatenate([np.zeros((N_ODORS, 1)), delta_pop_vector_matrix], axis=1)

        means_by_trial, stds_by_trial = np.mean(delta_pop_vector_matrix, axis=0), np.std(delta_pop_vector_matrix, axis=0)
        return delta_pop_vector_matrix, means_by_trial, stds_by_trial

    def drift_explained_by_pc(spike_counts):
        def calc_avg_drift_vector(spike_counts):
            drift_vectors = np.zeros((N_ODORS, TRIALS_PER_ODOR-1, N_NEURONS))
            for i in range(N_ODORS):
                in_odor_group = spike_counts[i*TRIALS_PER_ODOR : (i+1)*TRIALS_PER_ODOR]
                for j in range(TRIALS_PER_ODOR-1):
                    drift_vectors[i, j, :] = in_odor_group[j] - in_odor_group[j+1]
            return np.mean(drift_vectors, axis=0)
        
        pca = PCA(n_components=np.min(spike_counts.shape))
        splkes_pca = pca.fit_transform(spike_counts)

        pca_vars = pca.explained_variance_ratio_
        # effective_d = int(np.ceil(np.sum(pca_vars)**2 / np.sum(pca_vars**2))) 
        effective_d = 30

        spikes_pca_components = pca.components_[:effective_d, :]

        avg_drift_vector = calc_avg_drift_vector(spike_counts).T

        # pca components chosen such that drift projection >= 0
        gamma = np.abs(spikes_pca_components @ avg_drift_vector)  # (12, 1007) @ (1007, 24) = (12, 24)
        gamma /= np.linalg.norm(avg_drift_vector, axis=0) # divide by drift norm

        return gamma, pca_vars[:effective_d]/pca_vars[0]  # normalized to PC 1 variance
        

    spike_counts = gen_spike_counts(neurons, odor_starts, odors, sorted=True, times=[ODOR_START,ODOR_END]).T

    fig, axs = plt.subplots(2, 1, figsize=(9,10))
    pop_vector_angles, drift_means, drift_stds = drift_across_presentations(spike_counts, angle_between)
    pop_vector_corr_angles, drift_corr_means, drift_corr_stds = drift_across_presentations(spike_counts, corr_angle_between)

    pop_vector_angles_from_p0, drift_means_p0, drift_stds_p0 = drift_across_presentations(spike_counts, angle_between, compare_to_first=True)
    pop_vector_corr_angles_from_p0, drift_corr_means_p0, drift_corr_stds_p0 = drift_across_presentations(spike_counts, corr_angle_between, compare_to_first=True)

    drift_pair_space = np.arange(1, TRIALS_PER_ODOR)
    for i in range(N_ODORS):
        axs[0].scatter(drift_pair_space, pop_vector_angles[i, :], color='blue', s=15, alpha=0.6, label='drift' if i == 0 else None)
        axs[0].scatter(drift_pair_space, pop_vector_corr_angles[i, :], color='orange', s=15, alpha=0.6, label='mean-subtracted drift' if i == 0 else None)

        axs[1].scatter(ODOR_PRESENTATION_SPACE, pop_vector_angles_from_p0[i, :], color='blue', s=15, alpha=0.6, label='drift from first presentation' if i == 0 else None)
        axs[1].scatter(ODOR_PRESENTATION_SPACE, pop_vector_corr_angles_from_p0[i, :], color='orange', s=15, alpha=0.6, label='mean-subtracted drift from first presentation' if i == 0 else None)

    axs[0].set_xlabel('Presentation pair (n, n+1)')
    axs[0].errorbar(drift_pair_space, drift_means, yerr=drift_stds, fmt='o-', color='blue', ecolor='blue', capsize=3, label='± 1 std')
    axs[0].errorbar(drift_pair_space, drift_corr_means, yerr=drift_corr_stds, fmt='o-', color='orange', ecolor='orange', capsize=3)

    axs[1].set_xlabel('Drift from initial presentation (0, n)')
    axs[1].errorbar(ODOR_PRESENTATION_SPACE, drift_means_p0, yerr=drift_stds_p0, fmt='o-', color='blue', ecolor='blue', capsize=3, label='± std')
    axs[1].errorbar(ODOR_PRESENTATION_SPACE, drift_corr_means_p0, yerr=drift_corr_stds_p0, fmt='o-', color='orange', ecolor='orange', capsize=3, label='± std')

    for ax in axs:
        ax.set_ylabel('Mean drift (deg)')
        ax.set_xticks(drift_pair_space) 

    axs[0].legend()

    fig, axs = plt.subplots(2, 1, figsize=(9,10))
    gamma, explained_variances = drift_explained_by_pc(spike_counts)  # (12, 24)
    gamma_means = np.mean(gamma, axis=1)

    for i in range(TRIALS_PER_ODOR-1):
        axs[0].scatter(explained_variances, gamma[:, i], color='red', s=15, alpha=0.6, label='drift along each PC' if i == 0 else None)

    popt, _ = curve_fit(exp_func, explained_variances, gamma_means, p0=(1, 1, 0), maxfev=5000)
    x_fit = np.linspace(min(explained_variances), max(explained_variances), 200)
    axs[0].plot(x_fit, exp_func(x_fit, *popt), '-', color='black', label='exponential fit')

    # axs[0].errorbar(explained_variances, gamma_means, yerr=gamma_stds, fmt='o-', color='black', ecolor='black', capsize=3, label='gamma mean ± std')
    axs[0].set_xlabel('Variance explained, normalized to PC 1')
    axs[0].set_ylabel('Drift projection magnitude on each PC')
    axs[0].set_xscale('log')
    axs[0].set_title('Drift occurs most in the direction of top PCs')

    plt.show()

def main():

    global N_NEURONS

    if DIR == 'Dataset1':
        eventTimes = mat73.loadmat(DATA) 
        neurons = eventTimes['spikeTiming']['spikeTimesByUnit']
        odor_starts = np.array(eventTimes['stimTiming']['odorStarts'], dtype=float) # (200,)
        odors = np.array(eventTimes['stimTiming']['manifold_bottleids'][:,1], dtype=int) # (200,)

    elif DIR == 'Dataset2' and DIR == 'Dataset3':
        eventTimes = loadmat(DATA)
        neurons = eventTimes['spikeTimes'].squeeze()
        odor_starts = np.array(eventTimes['stimTimes'], dtype=float)
        odors = np.array(eventTimes['stimIDs'], dtype=int).ravel()

    else:
        eventTimes = mat73.loadmat(DATA)
        neurons = eventTimes['sua_units_s']
        odor_starts = np.array(eventTimes['stim_on'], dtype=float)
        odors = np.array(eventTimes['stim_id'], dtype=int).ravel()

    N_NEURONS = len(neurons)
    neuron = np.array(neurons[NEURON_NUMBER], dtype=float).flatten()

    # gen_M(neurons, dt=0.1) # 0.1 ms
    # gen_fig_a(neuron, tmax=20)
    # gen_fig_c(neurons, odor_starts, odors)
    # gen_fig_1f(neurons, odor_starts, odors)
    # plot_corr_pop_vector(neurons, odor_starts, odors)
    # trial_zscores(neurons, odor_starts, odors)
    # axs = delta(neurons, odor_starts, odors, operator.mul)
    # x(neurons, odor_starts, odors, axs)
    # svm(neurons, odor_starts, odors)
    # odor_trial_split(neurons, odor_starts, odors)
    # mean_pop_firing_rate(neurons, odor_starts, odors)
    # single_unit_regression(neurons, odor_starts, odors)

    # pre_post_time_series(neurons, odor_starts, odors)
    # cluster_pcorr_plots(neurons, odor_starts, odors)
    # cluster_pcorr_vary_k(neurons, odor_starts, odors)
    # cluster_in_pca_space(neurons, odor_starts, odors, k_list=[5, 8, 12], d_list=[50, 100, 200])
    # pcorr_pre_post_plotter(neurons, odor_starts, odors)
    # proj_evo_spon_pca(neurons, odor_starts, odors)
    plot_drift_across_presentations(neurons, odor_starts, odors)


if __name__ == "__main__":
    main()