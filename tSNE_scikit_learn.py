from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler

def main():
    sequences_data = pd.read_csv('inputs/sequences.csv')
    normal_data = pd.read_csv('inputs/normal_data.csv')
    x_data = normal_data.drop(columns=['label']).values
    labels_normal = normal_data['label'].values

    # Having a standardization for multi-featured data is always a good practice, though not required for this specific
    # data, since variance seems pretty similar between features. However, I still decided to have it here to make this
    # code adapted for non-normalized data as well.
    scaler = StandardScaler()
    x_data_scaled = scaler.fit_transform(x_data)

    # t-SNE for question 2, for normal_data.csv
    low_dim_embedding = TSNE(perplexity=10, max_iter=600, learning_rate=300,
                                      n_components=2, random_state=42).fit_transform(x_data_scaled)
                                      
    # t-SNE for question 3, for sequences.csv. Just one feature here, no need to standardize.
    sequences = sequences_data['sequence'].values
    labels = sequences_data['label'].values
    hamming_distance = compute_hamming_distance_matrix(sequences)
    low_dim_embedding_2 = TSNE(perplexity=100, max_iter=400, learning_rate=150,
                                          n_components=2, random_state=42).fit_transform(hamming_distance)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # First t-SNE plot
    scatter1 = axs[0].scatter(low_dim_embedding[:, 0], low_dim_embedding[:, 1], c=labels_normal, cmap='viridis')
    axs[0].set_title("t-SNE Embedding for normal data sklearn")
    axs[0].set_xlabel("Dimension 1")
    axs[0].set_ylabel("Dimension 2")
    plt.colorbar(scatter1, ax=axs[0])   
    unique_labels = sorted(set(labels_normal))
    colors = scatter1.cmap(scatter1.norm(unique_labels))
    legend_patches = [mpatches.Patch(color=colors[i], label=f'Label {unique_labels[i]}') for i in range(len(unique_labels))]
    axs[0].legend(handles=legend_patches, title="Labels")

    # Second t-SNE plot
    scatter2 = axs[1].scatter(low_dim_embedding_2[:, 0], low_dim_embedding_2[:, 1], c=labels, cmap='viridis')
    axs[1].set_title("t-SNE Embedding for kmers sklearn")
    axs[1].set_xlabel("Dimension 1")
    axs[1].set_ylabel("Dimension 2")
    plt.colorbar(scatter2, ax=axs[1])     
    unique_labels = sorted(set(labels))
    colors = scatter2.cmap(scatter2.norm(unique_labels))
    legend_patches = [mpatches.Patch(color=colors[i], label=f'Label {unique_labels[i]}') for i in range(len(unique_labels))]
    axs[1].legend(handles=legend_patches, title="Labels")
    # Show both plots
    plt.tight_layout()  
    plt.show()


def compute_hamming_distance_matrix(sequences: list[str]) -> np.ndarray:
    num_sequences = len(sequences)
    hamming_distance_matrix = np.zeros((num_sequences, num_sequences))
    for i in range(num_sequences):
        for j in range(num_sequences):
            hamming_distance_matrix[i, j] = sum(c1 != c2 for c1, c2 in zip(sequences[i], sequences[j]))
    return hamming_distance_matrix


if __name__ == "__main__":
    main()