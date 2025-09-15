import numpy as np

MATCH, INSERT, DELETE, SUBSTITUTE = 0, 1, 2, 3


def main():
    seq1 = "AGTACGCA"
    seq2 = "TATGC"
    nucleotide_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

    seq1_array = np.array([nucleotide_map[nuc] for nuc in seq1])
    seq2_array = np.array([nucleotide_map[nuc] for nuc in seq2])
    # DNAfull substitution matrix (also known as EDNAFULL)
    substitution_matrix = np.array([
        [5, -4, -4, -4],
        [-4, 5, -4, -4],
        [-4, -4, 5, -4],
        [-4, -4, -4, 5],])
    # Alternative popular substitution matrix
    # substitution_matrix = np.array([
    #     [1, -1, -1, -1],
    #     [-1, 1, -1, -1],
    #     [-1, -1, 1, -1],
    #     [-1, -1, -1, 1]])
    gap_opening_penalty = - 10  # linear gap penalty
    gap_extension_penalty = - 0.5  # affine ga penalty

    # This code has two main functions
    # 1) _needleman_wunsch_gotoh_kernel – our Needleman algorithm, gives us a map of optimal moves
    # across the matrix (changes)
    # 2) _reconstruct_alignment – a function to reconstruct alignments based on the map of optimal
    # moves (changes)

    # For both the purely linear case and the affine case with penalties – we use the same Needleman algorithm function.
    # But the function is written so that it figures out which calculation method to use depending on whether
    # an argument with an affine penalty was provided.

    # linear gap penalty
    scoring_1, change_1, local_alignment_score_1, global_alignment_score_1 = _needleman_wunsch_gotoh_kernel(
        seq1_array, seq2_array, substitution_matrix, gap_opening_penalty)
    align1_linear, align2_linear = _reconstruct_alignment(change_1, seq1_array, seq2_array, lambda x: "ATGC"[x],
                                                          scoring_1)
    print(scoring_1)
    print("Alignment result using linear gap penalty:")
    print("Aligned sequence 1:", align1_linear)
    print("Aligned sequence 2:", align2_linear)
    print("Local alignment score:", local_alignment_score_1)
    print("Global alignment score:", global_alignment_score_1)
    print('')

    # affine gap penalty
    scoring_2, change_2, deletes, inserts, local_alignment_score_2, global_alignment_score_2 =\
        _needleman_wunsch_gotoh_kernel(seq1_array, seq2_array, substitution_matrix, gap_opening_penalty,
                                       gap_extension_penalty)
    align1_affine, align2_affine = _reconstruct_alignment(change_2, seq1_array, seq2_array, lambda x: "ATGC"[x],
                                                          scoring_2)
    print("Alignment result using affine gap penalty:")
    print("Aligned sequence 1:", align1_affine)
    print("Aligned sequence 2:", align2_affine)
    print("Local alignment score:", local_alignment_score_2)
    print("Global alignment score:", global_alignment_score_2)


def _needleman_wunsch_gotoh_kernel(seq1, seq2, substitution_matrix, gap_opening: float, gap_extension: float = None):
    negative_infinitiy = -np.inf
    seq1_len = len(seq1)
    seq2_len = len(seq2)
    scores = np.zeros((seq1_len + 1, seq2_len + 1), dtype=np.float64)
    changes = np.zeros((seq1_len + 1, seq2_len + 1), dtype=np.uint8)

    use_gap_extension = gap_extension is not None

    if use_gap_extension:
        deletes = np.zeros((seq1_len + 1, seq2_len + 1), dtype=np.float64)
        inserts = np.zeros((seq1_len + 1, seq2_len + 1), dtype=np.float64)

    scores[0, 0] = 0
    for j in range(1, seq2_len + 1):
        scores[0, j] = gap_opening + ((j - 1) * gap_extension if use_gap_extension else (j - 1) * gap_opening)
        if use_gap_extension:
            deletes[0, j] = scores[0, j] + gap_opening + gap_extension

    for i in range(1, seq1_len + 1):
        scores[i, 0] = gap_opening + ((i - 1) * gap_extension if use_gap_extension else (i - 1) * gap_opening)
        if use_gap_extension:
            inserts[i, 0] = scores[i, 0] + gap_opening + gap_extension

        for j in range(1, seq2_len + 1):
            substitution = substitution_matrix[seq1[i - 1], seq2[j - 1]]
            replace = scores[i - 1, j - 1] + substitution
            if use_gap_extension:
                # Add a gap or extend a gap in seq2?
                delete = max(
                    scores[i - 1, j].item() + gap_opening,
                    deletes[i - 1, j].item() + gap_extension,
                )
                # Add a gap or extend a gap in seq1?
                insert = max(
                    scores[i, j - 1].item() + gap_opening,
                    inserts[i, j - 1].item() + gap_extension,
                )
            else:
                delete = scores[i - 1, j] + gap_opening  # Add a gap in seq2?
                insert = scores[i, j - 1] + gap_opening  # Add a gap in seq1?

            score = max(replace, delete, insert)
            scores[i, j] = score

            if use_gap_extension:
                deletes[i, j] = delete
                inserts[i, j] = insert

            if score == replace:
                changes[i, j] = MATCH if seq1[i - 1] == seq2[j - 1] else SUBSTITUTE
            elif score == delete:
                changes[i, j] = DELETE
            else:
                changes[i, j] = INSERT

        local_alignment_scoring = negative_infinitiy

        # Check the last row and column to find the maximum score
        # EMBOSS needle calculates the optimal score in the same way
        # We go through the last column and the last row and find the maximum value,
        # which essentially is the local alignment score
        for k in range(seq1_len + 1):
            local_alignment_scoring = max(local_alignment_scoring, scores[k, seq2_len].item())

        for m in range(seq2_len + 1):
            local_alignment_scoring = max(local_alignment_scoring, scores[seq1_len, m].item())
        global_alignment_scoring = scores[-1, -1]
    if use_gap_extension:
        return scores, changes, deletes, inserts, local_alignment_scoring, global_alignment_scoring
    else:
        return scores, changes, local_alignment_scoring, global_alignment_scoring


def _reconstruct_alignment(changes, seq1, seq2, code_to_char, scores):
    print(changes)
    align1, align2 = "", ""
    i, j = len(seq1), len(seq2)

    # Backtrack to reconstruct the alignment
    while i > 0 and j > 0:
        if changes[i, j] == MATCH or changes[i, j] == SUBSTITUTE:
            align1 = code_to_char(seq1[i - 1]) + align1
            align2 = code_to_char(seq2[j - 1]) + align2
            i -= 1
            j -= 1
        elif changes[i, j] == DELETE:
            align1 = code_to_char(seq1[i - 1]) + align1
            align2 = '-' + align2
            i -= 1
        elif changes[i, j] == INSERT:
            align1 = '-' + align1
            align2 = code_to_char(seq2[j - 1]) + align2
            j -= 1
    print(align1)
    print(align2)

    while i > 0:
        align1 = code_to_char(seq1[i - 1]) + align1
        align2 = '-' + align2
        i -= 1
    while j > 0:
        align1 = '-' + align1
        align2 = code_to_char(seq2[j - 1]) + align2
        j -= 1

    return align1, align2


if __name__ == "__main__":
    main()
