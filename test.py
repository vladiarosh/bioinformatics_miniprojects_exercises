import pickle
import numpy as np
import pandas as pd
from collections import Counter
from hmmlearn import hmm

# Reading training and test data, extracting signals for training
true_params = pd.read_csv('kmer_baseline_table.tsv', sep='\t', index_col=0)
with open('test_data_raw_signals.pkl', 'rb') as f:
    test_raw_signals = pickle.load(f)

with open('train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
baseline_mean = train_data['baseline_levels']
simulations = train_data['simulations']
kmers = train_data['kmers']
print(baseline_mean)

signal_arrays = [sim['raw_signal'] for sim in simulations]
signals_concatenated = np.concatenate(signal_arrays).reshape(-1, 1)


n_states = len(kmers)
lengths = []
for sim in simulations:
    length_of_array = len(sim['raw_signal'])
    lengths.append(length_of_array)
n_components = n_states

# Computing
variances = []
for kmer in kmers:
    kmer_signals = []
    for sim in simulations:
        states = np.array(sim['states'])
        raw_signals = np.array(sim['raw_signal'])
        kmer_signals.extend(raw_signals[states == kmers.index(kmer)])
    # Compute variance of signals for the k-mer
    kmer_variance = np.var(kmer_signals)
    variances.append(kmer_variance)

variances_reshaped = np.array(variances).reshape(-1, 1)
# print('Variances are', variances_reshaped)

# Testing 3 versions of startprob: descending, based on occurrence in training data and [1,0,0...0]
descending_probs = np.linspace(1, 0, n_states)

startprob = descending_probs / np.sum(descending_probs)

state_counts = Counter()

for simulation in simulations:
    state_counts.update(simulation['states'])

total_occurrences = sum(state_counts.values())
startprob_2 = np.array([state_counts[state] / total_occurrences for state in range(len(state_counts))])

# This one works best for prediction
startprob_3 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


# Defining model parameters
model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=50, random_state=42, algorithm='map',
                        verbose=True, init_params='')

# Defining transition probabilities matrix as a starting point for a model
prob = 1/3
model.transmat_ = np.array([[0.5,  0.5,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                          [prob, prob, prob,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                          [0,   prob, prob, prob,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                          [0,   0, prob, prob, prob,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                          [0,   0,   0,   prob, prob, prob,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                          [0,   0,   0,   0,   prob, prob, prob,  0,   0,   0,   0,   0,   0,   0,   0,   0],
                          [0,   0,   0,   0,   0,   prob, prob, prob,  0,   0,   0,   0,   0,   0,   0,  0],
                          [0,   0,   0,   0,   0,   0,   prob, prob, prob,  0,   0,   0,   0,   0,   0,   0],
                          [0,   0,   0,   0,   0,   0,   0,   prob, prob, prob,  0,   0,   0,   0,   0,   0],
                          [0,   0,   0,   0,   0,   0,   0,   0,   prob, prob, prob,  0,   0,   0,   0,   0],
                          [0,   0,   0,   0,   0,   0,   0,   0,   0,   prob, prob, prob,  0,   0,   0,   0],
                          [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   prob, prob, prob,  0,   0,   0],
                          [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   prob, prob, prob,  0,   0],
                          [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   prob, prob, prob,  0],
                          [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   prob, prob, prob],
                          [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0.5,   0.5]])

model.means_ = np.array(baseline_mean).reshape(-1, 1)
model.covars_ = variances_reshaped
model.startprob_ = startprob_3

model.fit(signals_concatenated, lengths=lengths)
inferred_transition_probs = model.transmat_
inferred_means = model.means_
inferred_variances = model.covars_

# Comparing inferred values to true data
comparison = []
for i, kmer in enumerate(kmers):
    true_mean = true_params.loc[kmer, 'level_mean']
    true_std = true_params.loc[kmer, 'level_stdv']
    inferred_mean = inferred_means[i, 0]
    inferred_std = np.sqrt(inferred_variances[i, 0])

    comparison.append({
        'k-mer': kmer,
        'inferred_mean': inferred_mean,
        'true_mean': true_mean,
        'mean_difference': abs(inferred_mean - true_mean),
        'inferred_std': inferred_std,
        'true_std': true_std,
        'std_difference': abs(inferred_std - true_std),
    })

comparison_df = pd.DataFrame(comparison)
comparison_df.to_csv('parameter_comparison.csv', index=False)

lengths_2 = []
for i in test_raw_signals:
    length = len(i)
    lengths_2.append(length)
test_data = np.concatenate(test_raw_signals).reshape(-1, 1)

predicted_kmers = model.predict(test_data, lengths=lengths_2)

print(predicted_kmers.tolist())

sublists = []
start = 0  # Start index for slicing
for length in lengths_2:
    end = start + length  # End index for slicing
    sublists.append(predicted_kmers[start:end])  # Slice the list
    start = end  # Update start index

print(sublists[0:4])
