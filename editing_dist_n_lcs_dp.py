import numpy as np

"""
Editing distance and longest common susequence (lcs) problem
The problems are very similar and can be derived from one another
with minor changes to the source code
"""

def edit_distance(seq1, seq2): 
    """
    Compute the solutions of all subproblems:  bottom-up
    Uses dynamic programming. Takes polynomial time
    O(n * m) complexity (time and space)

    input: numpy.array with chars
    """
    # find the length of the strings 
    n = seq1.shape[0] 
    m = seq2.shape[0]
    
    # base case, empty string
    if n == 0 or m == 0:
        return np.max(n, m)

    #init lookup table
    ed = np.zeros((n + 1, m + 1), dtype=int)
    ed[:, 0] = np.arange(n + 1)
    ed[0, :] = np.arange (m + 1)

    for i in range(1, n + 1): 
        for j in range(1, m + 1):
            # values are diff. by default (just because, removes an else statement)
            dist = 1 
            if seq1[i - 1] == seq2[j - 1]: 
                dist = 0
            ed[i, j] = np.min([ed[i - 1, j - 1] + dist, ed[i - 1, j] + 1, ed[i, j - 1] + 1]) 
    
    return ed[n, m]


def lcs(seq1, seq2): 
    """
    Longest Common Susequence (not to be confused with subtring)
    Compute the solutions of all subproblems:  bottom-up
    Uses dynamic programming. Takes polynomial time
    Subsequences do not need to ocupy contiguous positions
    O(n * m) complexity (time and space)
    """
    # find the length of the strings 
    n = seq1.shape[0] 
    m = seq2.shape[0]
    
    # base case, empty string
    if n == 0 or m == 0:
        return 0

    #init lookup table
    ed = np.zeros((n + 1, m + 1), dtype=int)
    #ed[:, 0] = np.arange(n + 1)
    #ed[0, :] = np.arange (m + 1)

    for i in range(1, n + 1): 
        for j in range(1, m + 1):
            # matching chars, increment seq. length
            if seq1[i - 1] == seq2[j - 1]: 
                ed[i, j] = ed[i - 1, j - 1] + 1
            # end of sub-sequence
            else: 
                ed[i, j] = np.max([ed[i - 1, j], ed[i, j - 1]]) 

    return ed[n, m]

def longest_common_substring(s1, s2):
    """
    Get the longest common substring in s1 and s2 
    """
    n = s1.shape[0]
    m = s2.shape[0]
    
    # init. aux. table
    ed = np.zeros((n + 1, m + 1))

    # length of longest common substring
    max_sub_str_len = 0
    # index of the last char of longest common substring in s1
    end_sub_str_i = 0
    # index of the last char of longest common substring in s1
    end_sub_str_j = 0

    for i in range(1, 1 + n):
        for j in range(1, 1 + m):
            # matched chars
            if s1[i - 1] == s2[j - 1]:
                ed[i][j] = ed[i - 1][j - 1] + 1
                # update longest_common_substring
                if ed[i][j] > max_sub_str_len:
                    # update max length
                    max_sub_str_len = ed[i][j]
                    # update end of subtring index
                    end_sub_str_i = i
                    end_sub_str_j = j
            # unmatched chars
            else:
                ed[i][j] = 0

    #lcsstr = s1[int(end_sub_str_i - max_sub_str_len): end_sub_str_i]
    
    return [int(end_sub_str_i - max_sub_str_len), end_sub_str_i]
