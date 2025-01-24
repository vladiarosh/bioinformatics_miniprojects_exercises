import pickle
import numpy as np
from tqdm import tqdm
from scipy.special import logsumexp


def main():
    # Reading training and test data, extracting signals for training
    with open('inputs/test_data_raw_signals.pkl', 'rb') as f:
        test_raw_signals = pickle.load(f)

    with open('inputs/train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    baseline_mean = train_data['baseline_levels']
    simulations = train_data['simulations']
    kmers = train_data['kmers']
    print(baseline_mean)

    lengths = []
    for sim in simulations:
        length_of_array = len(sim['raw_signal'])
        lengths.append(length_of_array)

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

    # Used the same statrprob as I used with hmmlean
    # This one works best for prediction
    startprob_3 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    startprob_3 = np.array(startprob_3).reshape(-1, 1)

    prob = 1/3
    transmat = np.array([[0.5,  0.5,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                    [prob, prob, prob,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                    [0,   prob, prob, prob,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                    [0,   0,   prob, prob, prob,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                    [0,   0,   0,   prob, prob, prob,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                    [0,   0,   0,   0,   prob, prob, prob,  0,   0,   0,   0,   0,   0,   0,   0,   0],
                    [0,   0,   0,   0,   0,   prob, prob, prob,   0,   0,   0,   0,   0,   0,   0,  0],
                    [0,   0,   0,   0,   0,   0,   prob, prob, prob,  0,   0,   0,   0,   0,   0,   0],
                    [0,   0,   0,   0,   0,   0,   0,   prob, prob, prob,  0,   0,   0,   0,   0,   0],
                    [0,   0,   0,   0,   0,   0,   0,   0,   prob, prob, prob,  0,   0,   0,   0,   0],
                    [0,   0,   0,   0,   0,   0,   0,   0,   0,   prob, prob, prob,  0,   0,   0,   0],
                    [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   prob, prob, prob,  0,   0,   0],
                    [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   prob, prob, prob,  0,   0],
                    [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   prob, prob, prob,  0],
                    [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  prob, prob, prob],
                    [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0.5,   0.5]])
    means = np.array(baseline_mean).reshape(-1, 1)

    all_sequences = []
    for read in tqdm(test_raw_signals):
        kmers = fb_algorithm(read, transmat, startprob_3, means, variances_reshaped)
        all_sequences.append(kmers)

    print(all_sequences[0:4])


# One of the first bottlenecks I encounterd, were very low probabilities, exceeding the border of floating point accuracy
# But I learned that it is a very common problem solved by turning all your calculations into log-space
# Luckily logsumexp trick already exists
# Using this trick I wrote a custom function for matrix dot-product in log-space
def log_dot_product(A_log, B_log):

    # Validate matrix dimensions for dot product
    assert A_log.shape[1] == B_log.shape[0], "Matrix dimensions do not align for multiplication."

    # Output matrix in log-space
    C_log = np.zeros((A_log.shape[0], B_log.shape[1]))

    # We first collect all the products and them perfrom their sum using logsumexp trick
    for i in range(A_log.shape[0]):
        for j in range(B_log.shape[1]):
            # Log-sum-exp for dot product of row i in A_log and column j in B_log
            C_log[i, j] = logsumexp(A_log[i, :] + B_log[:, j])

    return C_log


# Then I defined calculation of alpha and bets required for forward-backward algorithm
def forward_prob(x, pi_0, T, mu, var):
    x_num = x.shape[0]
    s_num = mu.shape[0]
    F_log = np.zeros((s_num, x_num))

    f = pi_0.T

    f_log = np.log(f)
    T_log = np.log(T)
    id_log = np.zeros((s_num, 1))

    for (i, x_i) in enumerate(x):
        emission_prob_log = -(x_i - mu) ** 2 / (2 * var) - np.log(np.sqrt(2 * np.pi * var))
        O_log = np.log(np.zeros((s_num, s_num)))
        np.fill_diagonal(O_log, emission_prob_log)

        f_log = log_dot_product(f_log, log_dot_product(T_log, O_log))
        f_log = f_log - log_dot_product(f_log, id_log)
        F_log[:, i] = f_log.flatten()

    return F_log


def backward_prob(x, T, mu, var):
    x_num = x.shape[0]
    s_num = mu.shape[0]
    B_log = np.zeros((s_num, x_num))

    b = np.ones((s_num, 1))

    b_log = np.log(b)
    T_log = np.log(T)
    id_log = np.zeros((1, s_num))

    for (i, x_i) in enumerate(np.flip(x)):
        b_log = b_log - log_dot_product(id_log, b_log)
        B_log[:, x_num - 1 - i] = b_log.flatten()

        emission_prob_log = -(x_i - mu) ** 2 / (2 * var) - np.log(np.sqrt(2 * np.pi * var))
        O_log = np.log(np.zeros((s_num, s_num)))
        np.fill_diagonal(O_log, emission_prob_log)

        b_log = log_dot_product(T_log, log_dot_product(O_log, b_log))

    return B_log


# Using our alpha and beta, we get the final array with prediction
def fb_algorithm(x, T, pi_0, mu, var):
    x_num = x.shape[0]
    s_num = mu.shape[0]
    S = np.zeros((s_num, x_num))
    id_log = np.zeros((1, s_num))

    F_log = forward_prob(x, pi_0, T, mu, var)
    B_log = backward_prob(x, T, mu, var)

    gamma_log = F_log + B_log
    gamma_log = gamma_log - log_dot_product(id_log, gamma_log)
    S = np.exp(gamma_log)

    s = np.argmax(S, axis=0)

    return s


if __name__ == "__main__":
    main()


