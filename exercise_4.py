import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler


def main():
    sequences_data = pd.read_csv('sequences.csv')
    normal_data = pd.read_csv('normal_data.csv')
    x_data = normal_data.drop(columns=['label']).values
    labels_normal = normal_data['label'].values

    # Having a standardization for multi-featured data is always a good practice, though not required for this specific
    # data, since variance seems pretty similar between features. However, I still decided to have it here to make this
    # code adapted for non-normalized data as well.
    scaler = StandardScaler()
    x_data_scaled = scaler.fit_transform(x_data)

    # t-SNE for question 2, for normal_data.csv
    low_dim_embedding, history = tsne(x_data_scaled, perplexity=10, iterations=600, learning_rate=300,
                                      embedding_dimensions=2, random_seed=42)

    scatter = plt.scatter(low_dim_embedding[:, 0], low_dim_embedding[:, 1], c=labels_normal, cmap='viridis')
    plt.colorbar()
    plt.title("t-SNE Embedding")
    unique_labels = sorted(set(labels_normal))
    colors = scatter.cmap(scatter.norm(unique_labels))
    legend_patches = [mpatches.Patch(color=colors[i], label=f'Label {unique_labels[i]}') for i in
                      range(len(unique_labels))]
    plt.legend(handles=legend_patches, title="Labels")
    plt.show()

    # # t-SNE for question 3, for sequences.csv. Just one feature here, no need to standardize.
    sequences = sequences_data['sequence'].values
    labels = sequences_data['label'].values
    hamming_distance = compute_hamming_distance_matrix(sequences)
    low_dim_embedding_2, history_2 = tsne(hamming_distance, perplexity=100, iterations=400, learning_rate=150,
                                          embedding_dimensions=2, random_seed=42)
    scatter = plt.scatter(low_dim_embedding_2[:, 0], low_dim_embedding_2[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title("t-SNE Embedding")
    unique_labels = sorted(set(labels))
    colors = scatter.cmap(scatter.norm(unique_labels))
    legend_patches = [mpatches.Patch(color=colors[i], label=f'Label {unique_labels[i]}') for i in
                      range(len(unique_labels))]
    plt.legend(handles=legend_patches, title="Labels")
    plt.show()


# 1. t-SNE implementation
def tsne(
    distance_matrix: np.ndarray,
    perplexity: int = 30,
    iterations: int = 1000,
    learning_rate: int = 200,
    exaggeration_factor: int = 4,
    embedding_dimensions: int = 2,
    random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    num_points = len(distance_matrix)
    pairwise_affinities = compute_pairwise_affinities(distance_matrix, perplexity)
    symmetric_affinities = symmetrize_affinities(pairwise_affinities)

    embeddings = np.zeros((iterations, num_points, embedding_dimensions))
    embeddings[0] = initialize_embedding(distance_matrix, embedding_dimensions, random_seed=random_seed)

    print("Optimizing Low Dimensional Embedding...")
    for t in range(1, iterations - 1):
        momentum = 0.5 if t < 250 else 0.8
        exaggeration = exaggeration_factor if t < 250 else 1

        low_dim_affinities = compute_low_dimensional_affinities(embeddings[t])
        gradient = compute_gradient(exaggeration * symmetric_affinities, low_dim_affinities, embeddings[t])
        embeddings[t + 1] = embeddings[t] - learning_rate * gradient + momentum * (embeddings[t] - embeddings[t - 1])

        if t % 50 == 0 or t == 1:
            cost = np.sum(symmetric_affinities * np.log(symmetric_affinities / low_dim_affinities))
            print(f"Iteration {t}: Cost = {cost}")
    print("Optimization Complete.")
    return embeddings[-1], embeddings


# 2. Function to find optimal sigma values
def find_optimal_sigma(differences: np.ndarray, index: int, perplexity: int) -> float:
    best_result = np.inf
    norms = np.linalg.norm(differences, axis=1)
    std_dev = np.std(norms)

    for sigma_candidate in np.linspace(0.01 * std_dev, 5 * std_dev, 200):
        probabilities = np.exp(-(norms ** 2) / (2 * sigma_candidate ** 2))
        probabilities[index] = 0
        normalized_probabilities = probabilities / (np.sum(probabilities) + np.finfo(float).eps)
        normalized_probabilities = np.maximum(normalized_probabilities, np.nextafter(0, 1))
        entropy = -np.sum(normalized_probabilities * np.log2(normalized_probabilities))
        difference = np.abs(np.log(perplexity) - entropy * np.log(2))
        if difference < best_result:
            best_result = difference
            best_sigma = sigma_candidate
    return best_sigma


# 3. Function to symmetrize affinities
def symmetrize_affinities(pairwise_affinities: np.ndarray) -> np.ndarray:
    num_points = len(pairwise_affinities)
    symmetric_affinities = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            symmetric_affinities[i, j] = (pairwise_affinities[i, j] + pairwise_affinities[j, i]) / (2 * num_points)
    epsilon = np.nextafter(0, 1)
    symmetric_affinities = np.maximum(symmetric_affinities, epsilon)
    return symmetric_affinities


# 4. Initialization
def initialize_embedding(data: np.ndarray, embedding_dimensions: int = 2, initialization_method: str = "random",
                         random_seed: int = None) -> np.ndarray:
    if random_seed is not None:
        np.random.seed(random_seed)
    if initialization_method == "random":
        return np.random.normal(loc=0, scale=1e-4, size=(len(data), embedding_dimensions))
    else:
        raise ValueError("Only random initialization is supported.")


# 5. Function to compute low-dimensional affinities
def compute_low_dimensional_affinities(embedding: np.ndarray) -> np.ndarray:
    num_points = len(embedding)
    low_dim_affinities = np.zeros((num_points, num_points))
    for i in range(num_points):
        differences = embedding[i] - embedding
        norms = np.linalg.norm(differences, axis=1)
        low_dim_affinities[i, :] = (1 + norms ** 2) ** (-1)
    np.fill_diagonal(low_dim_affinities, 0)
    low_dim_affinities /= low_dim_affinities.sum()
    epsilon = np.nextafter(0, 1)
    low_dim_affinities = np.maximum(low_dim_affinities, epsilon)
    return low_dim_affinities


# 6. Function for gradient computation
def compute_gradient(symmetric_affinities: np.ndarray, low_dim_affinities: np.ndarray,
                     embedding: np.ndarray) -> np.ndarray:
    num_points = len(symmetric_affinities)
    gradient = np.zeros((num_points, embedding.shape[1]))
    for i in range(num_points):
        differences = embedding[i] - embedding
        affinity_differences = symmetric_affinities[i, :] - low_dim_affinities[i, :]
        weight_matrix = (1 + np.linalg.norm(differences, axis=1)) ** (-1)
        gradient[i] = 4 * np.sum((affinity_differences[:, None] * weight_matrix[:, None]) * differences, axis=0)
    return gradient


# 7. Function to compute pairwise affinities matrix
def compute_pairwise_affinities(data: np.ndarray, perplexity: int = 10) -> np.ndarray:
    num_points = len(data)
    print("Computing Pairwise Affinities....")
    pairwise_affinities = np.zeros((num_points, num_points))

    for i in range(num_points):
        differences = data[i] - data
        sigma = find_optimal_sigma(differences, i, perplexity)
        norms = np.linalg.norm(differences, axis=1)
        pairwise_affinities[i, :] = np.exp(-(norms ** 2) / (2 * sigma ** 2))
        np.fill_diagonal(pairwise_affinities, 0)
        pairwise_affinities[i, :] = pairwise_affinities[i, :] / np.sum(pairwise_affinities[i, :])

    epsilon = np.nextafter(0, 1)
    pairwise_affinities = np.maximum(pairwise_affinities, epsilon)
    print("Completed Pairwise Affinities Matrix.\n")
    return pairwise_affinities


# 8. Function to compute Hamming distance matrix to get numerical representation of categorical input
def compute_hamming_distance_matrix(sequences: list[str]) -> np.ndarray:
    num_sequences = len(sequences)
    hamming_distance_matrix = np.zeros((num_sequences, num_sequences))
    for i in range(num_sequences):
        for j in range(num_sequences):
            hamming_distance_matrix[i, j] = sum(c1 != c2 for c1, c2 in zip(sequences[i], sequences[j]))
    return hamming_distance_matrix


if __name__ == "__main__":
    main()
