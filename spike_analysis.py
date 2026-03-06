import mat73
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import random
from scipy.io import loadmat
from scipy.sparse import lil_matrix, csr_matrix
from scipy.stats import mannwhitneyu
from itertools import combinations
import seaborn as sns

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

DIR = 'Dataset1' # DatasetRaleigh
FILE_NAME = 'eventTimes.mat' # APCdata.mat, eventTimes.mat
DATA = f'{DIR}/{FILE_NAME}'
NEURON_NUMBER = 9
RECORDING_LENGTH = 25200 # 7h

N_TRIALS = 200
N_ODORS = 8
TRIALS_PER_ODOR = int(N_TRIALS / N_ODORS)

N_ZSCORES = 504
N_BINS = 20

START = -2
ODOR_START = 0
ODOR_END = 4
END = 10  

BIN_WIDTH = (END - START) / N_BINS
LAST_BIN_BEFORE_ODOR = int(-START / BIN_WIDTH)
BINS = np.arange(START, END, BIN_WIDTH)
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

def gen_ccgs(neurons, lags, dt):
    """
    Generate all CCGs for spike matrix M with dimensions N x T (neurons x times)
    """
    M = gen_M(neurons, dt)


# limit pairwise conversions 
# ASK SAM how to compute auto correlogram, Andrew seems to remember he has some clever way of doing it
def gen_fig_a(neuron):
    # Computes pairwise intervals between stims

    n = len(neuron)
    i, j = np.triu_indices(n, k=1) # upper triangle indices (i < j)
    diffs = np.abs(neuron[i] - neuron[j])

    plt.figure(figsize=(10, 6))
    plt.hist(diffs, bins=500, edgecolor='black', alpha=0.7)

    plt.title('Histogram of All Pairwise Differences')
    plt.xlabel('Pairwise Difference (a - b)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
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
        neuron = np.array(neurons[i], dtype=float).flatten()
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
    plt.style.use('seaborn-talk')
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


def plot_corr_pop_vector(neurons, odor_starts, odors):
    corr_L1, corr_L2, p_corr, p_corr_sorted, p_corr_avgs, corrs_by_odor = population_vector_corrs(neurons, odor_starts, odors)

    # corr_L1 = corr_L1[:25, :25]
    # corr_L2 = corr_L2[:25, :25]
    # p_corr = p_corr[:25, :25]

    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle(f"{DIR} Sorted by Trial Number")

    corr_img = axs[0].imshow(corr_L1)
    fig.colorbar(corr_img, ax=axs[0])
    axs[0].set_title("L1 Corr")

    corr_img1 = axs[1].imshow(corr_L2, vmin=0.85, vmax=1)
    fig.colorbar(corr_img1, ax=axs[1])
    axs[1].set_title("L2 Corr")

    corr_img2 = axs[2].imshow(p_corr)
    fig.colorbar(corr_img2, ax=axs[2])
    axs[2].set_title("Pearson's Corr")

    # corr_img3 = axs[3].imshow(p_corr_shuffle)
    # fig.colorbar(corr_img3, ax=axs[3])
    # axs[3].set_title("Pearson's Corr Random Shuffle")

    axs[4].plot(p_corr_avgs)
    axs[4].set_xlabel("Trial Pair Corrs (1 --> pcorr(x1, x2))")
    axs[4].set_title("Pearson's Corr Between Consecutive Trials")

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
    for i, trial in enumerate([0, 49, 99, 149, 199]):
        sns.heatmap(z_scores[:, :, trial],ax=axs[i],cmap='RdBu_r',
        center=0,vmin=-2,vmax=2, cbar=False)
        axs[i].set_title(f'Trial {trial+1}')

    plt.tight_layout()
    plt.show()
    # plt.savefig("zscores_by_trial.png", dpi=300, bbox_inches="tight")


def delta(neurons, odor_starts, odors, consecutive=False):
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
            delta_ij[k, presentation_i, presentation_j, :] = spike_counts[:, i] - spike_counts[:, j]

    ij_pairs = [(int(i), int(j)) for i, j in combinations(ODOR_PRESENTATION_SPACE, 2)]
    _, axs = plt.subplots(2, 5, figsize=(20, 10))

    
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

                sliding_window_mask(2, group_novel_sliding_2, group_familiar_sliding_2, i, j)
                sliding_window_mask(3, group_novel_sliding_3, group_familiar_sliding_3, i, j)
                sliding_window_mask(4, group_novel_sliding_4, group_familiar_sliding_4, i, j)
            
            n_base = 2
            sliding_window_james(n_novel, 4, group_novel_james_4, group_familiar_james_4, ij_pairs, k, group1_rand, group2_rand, n_base)
            sliding_window_james(n_novel, 6, group_novel_james_6, group_familiar_james_6, ij_pairs, k, group1_rand, group2_rand, n_base)

        p_vals[n_novel] = mannwhitneyu(group_novel, group_familiar, alternative='two-sided')[1]
        if n_novel > n_base + 1:
            p_vals_rand[n_novel] = mannwhitneyu(group1_rand, group2_rand, alternative='two-sided')[1]
        p_vals_sliding[n_novel, 0] = mannwhitneyu(group_novel_sliding_2, group_familiar_sliding_2, alternative='two-sided')[1]
        p_vals_sliding[n_novel, 1] = mannwhitneyu(group_novel_sliding_3, group_familiar_sliding_3, alternative='two-sided')[1]
        p_vals_sliding[n_novel, 2] = mannwhitneyu(group_novel_sliding_4, group_familiar_sliding_4, alternative='two-sided')[1]
        if n_novel > n_base + 1:
            p_vals_sliding[n_novel, 3] = mannwhitneyu(group_novel_james_4, group_familiar_james_4, alternative='two-sided')[1]
            p_vals_sliding[n_novel, 4] = mannwhitneyu(group_novel_james_6, group_familiar_james_6, alternative='two-sided')[1]

        for i, arr in enumerate([group_novel, group_novel_sliding_2, group_novel_sliding_3, group_novel_sliding_4, group1_rand]):
            chunks = [arr[i:i+N_NEURONS] for i in range(0, len(arr), N_NEURONS)]
            l2_norms = [np.linalg.norm(chunk) for chunk in chunks]
            average_l2 = np.mean(l2_norms)
            delta_mags[n_novel, i] = average_l2

        if n_novel == 12 or n_novel == 17:
            y_axs = 2 if n_novel == 17 else 1
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

            axs[0, y_axs].plot(bin_centers, cdf_novel, label=f'Novel {n_novel}, James 6')
            axs[0, y_axs].plot(bin_centers, cdf_familiar, label=f'Familiar {n_novel}, James 6')
            axs[0, 3].plot(bin_centers, cdf_diff, label=f'Difference {n_novel}, Sum = {cdf_diff_value:.2f}')

            axs[0, y_axs].legend()
            axs[0, y_axs].legend()
            axs[0, 3].legend()

        # print(f'{n_novel}: novel: {len(group_novel)/1007}, avg: {np.mean(group_novel):.2f}, familiar: {len(group_familiar)/1007}, avg: {np.mean(group_familiar):.2f} p: {p_vals[n_novel]}')
        # print(f'{n_novel}: novel: {len(group1_rand)/1007}, avg: {np.mean(group1_rand):.2f}, familiar: {len(group2_rand)/1007}, avg: {np.mean(group2_rand):.2f} p: {p_vals_rand[n_novel]}')
        # print(f'{n_novel}: novel: {len(group_novel_sliding_4)/1007}, avg: {np.mean(group_novel_sliding_4):.2f}, familiar: {len(group_familiar_sliding_4)/1007}, avg: {np.mean(group_familiar_sliding_4):.2f} p: {p_vals_sliding[n_novel]}')
        # print(p_vals_sliding)

    n_novel_range = np.arange(0, len(p_vals))

    # axs[0, 0].plot(n_novel_range, p_vals, marker='o', label='δ Upper Triangular')
    # axs[0, 0].plot(n_novel_range, p_vals_sliding[:, 0], marker='x', label='δ Sliding window 2')
    # axs[0, 0].plot(n_novel_range, p_vals_sliding[:, 1], marker='x', label='δ Sliding window 3')
    # axs[0, 0].plot(n_novel_range, p_vals_sliding[:, 2], marker='x', label='δ Sliding window 4')
    axs[0, 0].plot(n_novel_range, p_vals_sliding[:, 3], marker='x', label='δ 4 Sliding window James')
    axs[0, 0].plot(n_novel_range, p_vals_sliding[:, 4], marker='x', label='δ 6 Sliding window James')
    axs[0, 0].plot(n_novel_range, p_vals_rand, marker='s', label='δ Random')
    axs[0, 0].set_yscale('log')   
    # axs[0, 0].set_ylim(1e-22, 1) 
    axs[0, 0].set_xlabel('Number of novel odors')
    axs[0, 0].set_ylabel('p')
    axs[0, 0].set_title(f'Mann-Whitney: {DIR}')
    axs[0, 0].legend(fontsize=8)
    axs[0, 0].grid(True)

    axs[0, 4].plot(n_novel_range, delta_mags[:, 0], marker='o', label='δ Upper Triangular')
    axs[0, 4].plot(n_novel_range, delta_mags[:, 1], marker='x', label='δ Sliding window 2')
    axs[0, 4].plot(n_novel_range, delta_mags[:, 2], marker='x', label='δ Sliding window 3')
    axs[0, 4].plot(n_novel_range, delta_mags[:, 3], marker='x', label='δ Sliding window 4')
    axs[0, 4].plot(n_novel_range, delta_mags[:, 4], marker='s', label='δ Random')
    axs[0, 4].set_xlabel('Number of novel odors')
    axs[0, 4].set_ylabel('|δ|₂')
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

    
    for n_novel in range(1, TRIALS_PER_ODOR-1):
        group_novel, group_familiar = [], []
        group_novel_sliding_2, group_familiar_sliding_2 = [], []
        group_novel_sliding_3, group_familiar_sliding_3 = [], []
        group_novel_sliding_4, group_familiar_sliding_4 = [], []
        group1_rand, group2_rand = [], []
        for k in range(N_ODORS):
            for i in range(TRIALS_PER_ODOR):
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

            axs[1, y_axs].plot(bin_centers, cdf_novel, label=f'Novel {n_novel}, UT')
            axs[1, y_axs].plot(bin_centers, cdf_familiar, label=f'Familiar {n_novel}, UT')
            axs[1, 3].plot(bin_centers, cdf_diff, label=f'Difference {n_novel}, Sum = {cdf_diff_value:.2f}')
            
            axs[1, y_axs].legend()
            axs[1, y_axs].legend()
            axs[1, 3].legend()

        # print(f'{n_novel}: novel: {len(group_novel)/1007}, avg: {np.mean(group_novel):.2f}, familiar: {len(group_familiar)/1007}, avg: {np.mean(group_familiar):.2f} p: {p_vals[n_novel]}')
        # print(f'{n_novel}: novel: {len(group1_rand)/1007}, avg: {np.mean(group1_rand):.2f}, familiar: {len(group2_rand)/1007}, avg: {np.mean(group2_rand):.2f} p: {p_vals_rand[n_novel]}')
        # print(f'{n_novel}: novel: {len(group_novel_sliding_4)/1007}, avg: {np.mean(group_novel_sliding_4):.2f}, familiar: {len(group_familiar_sliding_4)/1007}, avg: {np.mean(group_familiar_sliding_4):.2f} p: {p_vals_sliding[n_novel]}')
        # print(p_vals_sliding)

    n_novel_range = np.arange(0, len(p_vals))

    axs[1, 0].plot(n_novel_range, p_vals, marker='o', label='X Upper Triangular')
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

    axs[1, 4].plot(n_novel_range, x_mags[:, 0], marker='o', label='X Upper Triangular')
    axs[1, 4].plot(n_novel_range, x_mags[:, 1], marker='x', label='x Sliding window 2')
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
    spike_counts_spontaneous = gen_spike_counts(neurons, odor_starts, odors, sorted=True, times=[-12,-2]).T

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

    _, axs = plt.subplots(1, 4, figsize=(20, 5))
    plt.subplots_adjust(wspace=0.35)

    pca_spont = PCA(n_components=2)
    X_spontaneous_pca = pca_spont.fit_transform(spike_counts_spontaneous)

    pca_odor = PCA(n_components=2)
    X_pca = pca_odor.fit_transform(spike_counts)

    spontaneous_variance = pca_spont.explained_variance_ratio_ * 100
    odor_evoked_variance = pca_odor.explained_variance_ratio_ * 100

    axs[0].plot(pca_odor.components_[0], label=f'PC1 ({odor_evoked_variance[0]:.1f}%)')
    axs[0].plot(pca_odor.components_[1], label=f'PC2 ({odor_evoked_variance[1]:.1f}%)')

    axs[0].plot(pca_spont.components_[0], label=f'Spon. PC1 ({spontaneous_variance[0]:.1f}%)')
    axs[0].plot(pca_spont.components_[1], label=f'Spon. PC2 ({spontaneous_variance[1]:.1f}%)')

    axs[0].set_xlabel("Neuron")
    axs[0].set_ylabel("PC Loading Weight")
    axs[0].legend()
    # axs[0].plot(ODOR_PRESENTATION_SPACE, accuracy_linear*100, label='Linear SVM')
    # axs[0].axhline(50, color='red', linestyle='--', label='Baseline')
    # axs[0].set_xlabel('# Novel Presentations')
    # axs[0].set_ylabel('Accuracy')
    # axs[0].set_xlim(1, TRIALS_PER_ODOR-2)
    # axs[0].set_ylim(40, 100)
    # axs[0].set_title(f'{DIR}: Spontaneous Activity Classification Accuracy')
    # axs[0].legend()

    # SVM visualization
    for i, n_novel in enumerate([3, 10, 17]):
        Y = np.tile(np.r_[np.zeros(n_novel), np.ones(25-n_novel)], TRIALS_PER_ODOR)[:N_TRIALS].astype(int)
        
        # pca = PCA(n_components=2)
        # X_spontaneous_pca = pca.fit_transform(spike_counts_spontaneous)
        # X_2d = StandardScaler().fit_transform(X_spontaneous_pca)

        # fit odor evoked activity to spontaneous PCs
        X_odor_pca_spont = pca_spont.transform(spike_counts_spontaneous)


        padding = 0.5
        x_min, x_max = X_odor_pca_spont[:, 0].min() - padding, X_odor_pca_spont[:, 0].max() + padding
        y_min, y_max = X_odor_pca_spont[:, 1].min() - padding, X_odor_pca_spont[:, 1].max() + padding
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                            np.linspace(y_min, y_max, 500))
        
        clf_pca_linear = SVC(kernel='linear', C=0.1)
        clf_pca_linear.fit(X_odor_pca_spont, Y)
        Z_linear = clf_pca_linear.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z_linear = Z_linear.reshape(xx.shape)

        axs[i+1].contourf(xx, yy, Z_linear, levels=[-np.inf, 0, np.inf], colors=['#AAAAFF', '#FFAAAA'], alpha=0.3)
        axs[i+1].contour(xx, yy, Z_linear, levels=[0], colors='k', linewidths=2, label=f'Z_min = {Z_linear.min()}, Z_max = {Z_linear.max()}')
        scatter = axs[i+1].scatter(X_odor_pca_spont[:, 0], X_odor_pca_spont[:, 1], c=Y, cmap='coolwarm', edgecolor='k', s=40)
        handles, _ = scatter.legend_elements()
        axs[i+1].legend(handles, ['Novel', 'Familiar'], loc="upper right")
        axs[i+1].set_title(f"2D Linear SVM, n_novel = {n_novel}\nExplained Variance: {np.sum(pca_spont.explained_variance_ratio_)*100:.1f}%")
        axs[i+1].set_xlabel("Spontaneous PC 1")
        axs[i+1].set_ylabel("Spontaneous PC 2")
    
    plt.show()


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
    spike_counts = gen_spike_counts(neurons, odor_starts, odors, sorted=True, times=[ODOR_START,ODOR_END])
    X = spike_counts.reshape(1007, 8, 25)

    mean = X.mean(axis=(0,1))
    std  = X.std(axis=(0,1))

    plt.errorbar(range(1,26), mean, yerr=std, fmt='o')
    plt.style.use('seaborn-v0_8-talk')
    plt.xlabel("# of Odor Presentations")
    plt.ylabel("Firing Rate")
    plt.title(f"{DIR}: Mean Odor Evoked Response Firing Rate")
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
    # gen_fig_a(neurons)
    # gen_fig_c(neurons, odor_starts, odors)
    # gen_fig_1f(neurons, odor_starts, odors)
    plot_corr_pop_vector(neurons, odor_starts, odors)
    # trial_zscores(neurons, odor_starts, odors)
    # axs = delta(neurons, odor_starts, odors)
    # x(neurons, odor_starts, odors, axs)
    # svm(neurons, odor_starts, odors)
    # odor_trial_split(neurons, odor_starts, odors)
    # mean_pop_firing_rate(neurons, odor_starts, odors)
    


if __name__ == "__main__":
    main()