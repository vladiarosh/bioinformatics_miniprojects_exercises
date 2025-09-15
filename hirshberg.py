import numpy as np
from needleman_wunsch_gotoh_modified import _needleman_wunsch_gotoh_kernel
from needleman_wunsch_gotoh_modified import _reconstruct_alignment


def main():
    seq1 = "ATCCGAGTT"
    seq2 = "ATCAGTC"
    nucleotide_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

    seq1_array = np.array([nucleotide_map[nuc] for nuc in seq1])
    seq2_array = np.array([nucleotide_map[nuc] for nuc in seq2])

    substitution_matrix = np.array([
        [5, -4, -4, -4],
        [-4, 5, -4, -4],
        [-4, -4, 5, -4],
        [-4, -4, -4, 5], ])

    gap_opening_penalty = - 10
    a, b = hirschberg(seq1_array, seq2_array, substitution_matrix, gap_opening_penalty)
    print("Alignment result using linear gap penalty:")
    print(a)
    print(b)


def forwards(x, y, substitution_matrix, gap_opening):
    scores, changes, local_alignment_scoring, global_alignment_scoring = _needleman_wunsch_gotoh_kernel(
        x, y, substitution_matrix, gap_opening
    )
    return scores[-1]


def backwards(x, y, substitution_matrix, gap_opening):
    scores, changes, local_alignment_scoring, global_alignment_scoring = _needleman_wunsch_gotoh_kernel(
        x[::-1], y[::-1], substitution_matrix, gap_opening
    )
    return scores[-1]


def hirschberg(seq1, seq2, substitution_matrix, gap_opening):
    n, m = len(seq1), len(seq2)
    if n < 2 or m < 2:
        scores, changes, _, _ = _needleman_wunsch_gotoh_kernel(seq1, seq2, substitution_matrix, gap_opening)
        align1_linear, align2_linear = _reconstruct_alignment(changes, seq1, seq2, lambda x: "ATGC"[x], scores)
        return align1_linear, align2_linear
    else:
        F = forwards(seq1[:n // 2], seq2, substitution_matrix, gap_opening)
        B = backwards(seq1[n // 2:], seq2, substitution_matrix, gap_opening)

        partition = F + B[::-1]
        cut = np.argmax(partition)
        align1_left, align2_left = hirschberg(seq1[:n // 2], seq2[:cut], substitution_matrix, gap_opening)
        align1_right, align2_right = hirschberg(seq1[n // 2:], seq2[cut:], substitution_matrix, gap_opening)

        align1 = align1_left + align1_right
        align2 = align2_left + align2_right
        return align1, align2


if __name__ == "__main__":
    main()
