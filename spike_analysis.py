import mat73
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.io import loadmat
from scipy.sparse import lil_matrix, csr_matrix
from scipy.stats import mannwhitneyu, ks_2samp, ttest_ind
from itertools import combinations
import seaborn as sns

DIR = 'Dataset1'
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

def trial_odor_pairs(neuron, odor_starts, odors, start, end, sorted=True):
    trials = []
    for t in odor_starts:
        cond = (neuron > t + start) & (neuron < t + end)
        trials.append(neuron[cond] - t)

    # sort rasters based on odor number
    if sorted:
        sorted_odor_ind = np.argsort(odors)
        odors = odors[sorted_odor_ind]
        trials = [trials[i] for i in sorted_odor_ind]

    return trials, odors

def neuron_spike_rates(trials):
    neuron_spike_rate = np.zeros((N_TRIALS, N_BINS-1), dtype=float)
    for i, trial in enumerate(trials):
        counts, edges = np.histogram(trial, bins=BINS)
        neuron_spike_rate[i] = counts

    groups = neuron_spike_rate.reshape(N_ODORS, TRIALS_PER_ODOR, -1) # 8 x (25, N_BINS)
    spike_rate_odor_mu = np.mean(groups, axis=1) / (TRIALS_PER_ODOR * BIN_WIDTH) # (8, N_BINS)

    return spike_rate_odor_mu, edges


def gen_fig_c(neuron, odor_starts, odors):
    num_trials = len(odor_starts)
    trials, odors = trial_odor_pairs(neuron, odor_starts, odors, START, END)

    _, ax = plt.subplots(figsize=(9, 9)) 
    for i, spikes in enumerate(trials):
        color = COLORS[odors[i] - 1]
        
        ax.eventplot(spikes,
                    orientation='horizontal',
                    colors=color,
                    lineoffsets=i + 1,
                    linelengths=0.9,
                    linewidths=1.0)
    ax.set_xlim(START, END)
    ax.set_ylim(0.5, num_trials + 0.5)
    tick_positions = np.arange(1, num_trials + 1, TRIALS_PER_ODOR)   # 1, 26, 51, ..., 176
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([f'{k}' for k in tick_positions], fontsize=9)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trial/Odor')
    ax.set_title('Spike Raster of Odor Presentation at 0s')
    ax.grid(True, axis='x', alpha=0.25, linestyle=':')
    plt.tight_layout()
    plt.show()

    spike_rate_odor_mu, edges = neuron_spike_rates(trials)

    for i in range(N_ODORS):
        plt.plot(edges[:-1], spike_rate_odor_mu[i])
    plt.xlabel('Time from odor onset (s)')
    plt.ylabel('Firing Rate (Hz)')
    plt.show()

def gen_fig_1f(neurons, odor_starts, odors):
    z_scores = np.zeros((N_ZSCORES, N_BINS-1), dtype=float)

    for i in range(N_ZSCORES // N_ODORS):
        neuron = np.array(neurons[i], dtype=float).flatten()
        trials, odors = trial_odor_pairs(neuron, odor_starts, odors, START, END)
        spike_rate_odor_mu, _ = neuron_spike_rates(trials)

        spikes_without_odor = spike_rate_odor_mu[:, :LAST_BIN_BEFORE_ODOR]
        spike_rate_mu = np.mean(spikes_without_odor)
        spike_rate_std = np.std(spikes_without_odor)

        for j in range(N_ODORS):
            idx = i*N_ODORS + j
            z_scores[idx] = (spike_rate_odor_mu[j] - spike_rate_mu) / spike_rate_std # (1, N_BINS)
    
    z_scores = z_scores[(-np.nansum(z_scores, axis=1)).argsort()]

    plt.figure(figsize=(5, 8))
    sns.heatmap(z_scores, cmap='RdBu_r', center=0, 
                vmin=-3, vmax=3,
                cbar_kws={'label': 'Z-score'})
    plt.title('Z-score Heatmap')
    plt.xlabel('Bin')
    plt.ylabel('Neuron Odor Pairs')
    plt.tight_layout()
    plt.show()

def gen_spike_counts(neurons, odor_starts, odors, sorted=True, times=[]):
    """
    Population vectors across trials N_NEURONS x N_TRIALS
    If times = []: total spike count during 4 seconds of odor presentation
    If times = [t_start, t_end]: total spike count between t_start, t_end
    """

    t_start = times[0] if times else ODOR_START
    t_end = times[1] if times else ODOR_END

    spike_counts = np.zeros((N_NEURONS, N_TRIALS))
    for i in range(N_NEURONS):
        neuron = np.array(neurons[i], dtype=float).flatten()
        trials, odors = trial_odor_pairs(neuron, odor_starts, odors, START, END, sorted)
        for j in range(N_TRIALS):
            spike_counts[i,j] = sum(1 for t in trials[j] if t_start < t < t_end) 
    
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
        trials, _ = trial_odor_pairs(neuron, odor_starts, odors, START, END, sorted=False)
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

    _, axs = plt.subplots(20, 10, figsize=(30, 100))
    axs = axs.flatten()
    for i in range(200):
        print(i)
        sns.heatmap(z_scores[:, :, i],ax=axs[i],cmap='RdBu_r',
        center=0,vmin=-3,vmax=3,cbar=(i == 0), cbar_kws={'label': 'Z-score'})
        axs[i].set_title(f'Trial {i+1}')

    plt.tight_layout()
    # plt.show()
    plt.savefig("zscores_by_trial.png", dpi=300, bbox_inches="tight")


def delta(neurons, odor_starts, odors, consecutive=False):
    # compute p value separately per odor
    # understand how p value is computed
    # make histogram of the sets of numbers being used of p values, super impose them
    # i-j <= number --> sliding window, see if anything is chaning, odor specific trials
    # test p values with random shuffle and plot that instead (randomly draw pairs of i,j. populate two distributions with random draws)
    # do that enough times to get a distribution of p values
    """
    Generate delta_ij, the population vector difference between pairs of within-odor trials
    Determine if odor presentations {1...n_novel} are drawn from different distribution than {n_novel+1...25}
    """
    delta_ij = np.zeros((N_ODORS, TRIALS_PER_ODOR, TRIALS_PER_ODOR, N_NEURONS))
    odor_idxs = np.zeros((N_ODORS, TRIALS_PER_ODOR), dtype=int)
    spike_counts = gen_spike_counts(neurons, odor_starts, odors, sorted=False, times=[ODOR_START,ODOR_END])
    odor_presentation_space = np.arange(0, 25)
    p_vals = np.zeros(TRIALS_PER_ODOR-1)
    p_vals_x = np.zeros(TRIALS_PER_ODOR-1)

    for k in range(N_ODORS):
        odor_idxs[k] = np.where(odors == k+1)[0]
        ij_pairs = [(int(i), int(j)) for i, j in combinations(odor_idxs[k], 2)]
        for i,j in ij_pairs:
            presentation_i = np.where(odor_idxs[k] == i)[0][0]
            presentation_j = np.where(odor_idxs[k] == j)[0][0]
            delta_ij[k, presentation_i, presentation_j, :] = spike_counts[:, i] - spike_counts[:, j]

    ij_pairs = [(int(i), int(j)) for i, j in combinations(odor_presentation_space, 2)]
    for n_novel in range(1, TRIALS_PER_ODOR-1):
        group_novel, group_familiar = [], []
        group_novel_x, group_familiar_x = [], []
        for k in range(N_ODORS):
            if consecutive: 
                ij_pairs = np.stack([odor_presentation_space[:-1], odor_presentation_space[1:]], axis=1)
            for i,j in ij_pairs:
                if i <= n_novel and j <= n_novel:
                    group_novel.extend(delta_ij[k, i, j, :])
                    group_novel_x.extend(spike_counts[:, i])
                else:
                    group_familiar.extend(delta_ij[k, i, j, :])
                    group_familiar_x.extend(spike_counts[:, i])
        
        p_vals[n_novel] = mannwhitneyu(group_novel, group_familiar, alternative='two-sided')[1]
        p_vals_x[n_novel] = mannwhitneyu(group_novel_x, group_familiar_x, alternative='two-sided')[1]
        _, kp = ks_2samp(group_novel, group_familiar)
        _, ttest = ttest_ind(group_novel, group_familiar, equal_var=False)
        print(f'd{n_novel}: novel: {len(group_novel)/1007}, avg: {np.mean(group_novel):.2f}, familiar: {len(group_familiar)/1007}, avg: {np.mean(group_familiar):.2f} p: {p_vals[n_novel]}, ks: {kp}, ttest: {ttest}')
        print(f'x{n_novel}: novel: {len(group_novel_x)/1007}, avg: {np.mean(group_novel_x):.2f}, familiar: {len(group_familiar_x)/1007}, avg: {np.mean(group_familiar_x):.2f} p: {p_vals_x[n_novel]}')

    # -------------- CONSEC. TRIALS -----------------------
    delta_ij_con = np.zeros((N_NEURONS, N_TRIALS))
    p_vals_consec = np.zeros(TRIALS_PER_ODOR-1)
    for i in range(N_TRIALS-1):
        delta_ij_con[:, i] = spike_counts[:, i] - spike_counts[:, i+1]   
    for n_novel in range(1, TRIALS_PER_ODOR-1):
        group_novel = delta_ij_con[:, :8*n_novel].flatten()
        group_familiar =  delta_ij_con[:, 8*n_novel:].flatten()
        p_vals_consec[n_novel] = mannwhitneyu(group_novel, group_familiar, alternative='two-sided')[1]
        print(f'{n_novel}: novel: {len(group_novel)/1007}, avg: {np.mean(group_novel):.2f}, familiar: {len(group_familiar)/1007}, avg: {np.mean(group_familiar):.2f} p: {p_vals_consec[n_novel]}')
    
    n_novel_range = np.arange(0, len(p_vals))
    plt.figure(figsize=(8,4))
    plt.plot(n_novel_range, p_vals, marker='o', label='δ Consecutive odor pairs')
    plt.plot(n_novel_range, p_vals_consec, marker='s', label='δ Consecutive trial pairs')
    plt.plot(n_novel_range, p_vals_x, marker='x', label='x Consecutive odor pairs')
    plt.yscale('log')           # optional if p-values span many orders
    plt.xlabel('Number of novel odors')
    plt.xlim(1,25)
    plt.ylabel('p Mann-Whitney')
    plt.title('Comparison of p-values')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    global N_NEURONS

    if DIR == 'Dataset1':
        eventTimes = mat73.loadmat(DATA) 
        neurons = eventTimes['spikeTiming']['spikeTimesByUnit']
        odor_starts = np.array(eventTimes['stimTiming']['odorStarts'], dtype=float) # (200,)
        odors = np.array(eventTimes['stimTiming']['manifold_bottleids'][:,1], dtype=int) # (200,)

    else:
        eventTimes = loadmat(DATA)
        neurons = eventTimes['spikeTimes'].squeeze()
        odor_starts = np.array(eventTimes['stimTimes'], dtype=float)
        odors = np.array(eventTimes['stimIDs'], dtype=int).ravel()

    
    N_NEURONS = len(neurons)

    neuron = np.array(neurons[NEURON_NUMBER], dtype=float).flatten()

    # gen_M(neurons, dt=0.1) # 0.1 ms
    # gen_fig_a(neurons)
    # gen_fig_c(neuron, odor_starts, odors)
    # gen_fig_1f(neurons, odor_starts, odors)
    # plot_corr_pop_vector(neurons, odor_starts, odors)
    # trial_zscores(neurons, odor_starts, odors)
    delta(neurons, odor_starts, odors, consecutive=True)


if __name__ == "__main__":
    main()