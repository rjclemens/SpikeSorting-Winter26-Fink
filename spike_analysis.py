import mat73
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FILE = 'Dataset/eventTimes.mat'
NEURON_NUMBER = 9

NEURON_COUNT = 1007
N_TRIALS = 200
N_ODORS = 8
TRIALS_PER_ODOR = int(N_TRIALS / N_ODORS)

N_ZSCORES = 504
N_BINS = 80

START = -2
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

# limit pairwise conversions 
# ASK SAM how to compute auto correlogram, Andrew seems to remember he has some clever way of doing it
def gen_fig_a(neuron_spike_times):
    # Computes pairwise intervals between stims

    n = len(neuron_spike_times)
    i, j = np.triu_indices(n, k=1) # upper triangle indices (i < j)
    diffs = np.abs(neuron_spike_times[i] - neuron_spike_times[j])

    plt.figure(figsize=(10, 6))
    plt.hist(diffs, bins=500, edgecolor='black', alpha=0.7)

    plt.title('Histogram of All Pairwise Differences')
    plt.xlabel('Pairwise Difference (a - b)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def trial_odor_pairs(neuron_spike_times, odor_starts, odors, START, END):
    trials = []
    for t in odor_starts:
        cond = (neuron_spike_times > t + START) & (neuron_spike_times < t + END)
        trials.append(neuron_spike_times[cond] - t)

    # sort rasters based on odor number
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


def gen_fig_c(neuron_spike_times, odor_starts, odors):
    num_trials = len(odor_starts)
    trials, odors = trial_odor_pairs(neuron_spike_times, odor_starts, odors, START, END)

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
        neuron_spike_times = np.array(neurons[i], dtype=float).flatten()
        trials, odors = trial_odor_pairs(neuron_spike_times, odor_starts, odors, START, END)
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


def main():
    eventTimes = mat73.loadmat(FILE)  

    neuron_stims_struct = eventTimes['spikeTiming']['spikeTimesByUnit']
    odor_starts = np.array(eventTimes['stimTiming']['odorStarts'], dtype=float) # (200,)
    odors = np.array(eventTimes['stimTiming']['manifold_bottleids'][:,1], dtype=int) # (200,)

    neuron = np.array(neuron_stims_struct[NEURON_NUMBER], dtype=float).flatten()

    # gen_fig_a(neuron)
    # gen_fig_c(neuron, odor_starts, odors)
    gen_fig_1f(neuron_stims_struct, odor_starts, odors)


if __name__ == "__main__":
    main()