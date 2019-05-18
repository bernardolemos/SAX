import os
import argparse
import numpy as np
import pandas as pd
from time import time
from scipy.stats import norm
from scipy.spatial.distance import euclidean
from editing_dist_n_lcs_dp import edit_distance
from editing_dist_n_lcs_dp import lcs

#global variables
# BREAK_POINTS = []
# LOOKUP_TABLE = []

# TODO BUILD CLASS
# TODO find optimal VOCAB_SIZE & PAA_SIZE OR WINDOW_SIZE
# TODO compare multiple series
# TODO find motifs (cycles)

def matrix_to_df(cols, matrix):
    """
    Convert matrix of time series to pd.DataFrame
    """
    df = pd.DataFrame()

    for i in range(len(cols)):
      df[cols[i]] = matrix[i]  

    return df

def znorm(ts):
    """
    Standardize data
    """
    return (ts - np.mean(ts)) / np.std(ts)

def ts2paa(ts, paa_size):
    """
    PAA algorithm implementation
    The conde is inpired on the R SAX package code. For non-equidivisible PAA interval a weighted sum is applied,
    The R package the weighted sum imlementationh has O(n * paa_size) complexity, instead this function has O(n) complexity. 
    """
    # convert ts to a single value
    if paa_size == 1:
        return np.array(np.mean(ts))
    # use all ts' values
    elif paa_size == ts.shape[0]:
        return ts
    # series' length is divisible by paa split 
    elif ts.shape[0] % paa_size == 0:
        ts_split = np.reshape(ts, (paa_size, ts.shape[0]//paa_size))
        return np.mean(ts_split, 1)
    # ts' length is not divisible by paa split
    # O(ts.shape[0]) complexity instead of O(ts.shape[0] * paa_size)
    else:
        ts_paa = np.zeros(paa_size)
        carry = 0
        n_vals = 0
        paa_id = 0
        weight = paa_size
        for i in range(ts.shape[0]):
            # update number of computed values
            n_vals += paa_size   
            # set value's weight
            weight = paa_size
            # compute sum
            ts_paa[paa_id] += weight * ts[i] + carry
            # set carry
            carry = 0
            # verify integrety => update `weight` and compute `carry`
            # update sum
            if n_vals > ts.shape[0]:
                # update weight to remove excess sum
                weight = n_vals - ts.shape[0]
                # remove excess
                ts_paa[paa_id] -= weight * ts[i]
                #compute paa value
                ts_paa[paa_id] = ts_paa[paa_id] / ts.shape[0]
                # update paa_id and aux. values
                paa_id += 1
                n_vals = weight
                carry = weight * ts[i]

        return ts_paa

def get_breakpoints(vocab_size):
    """
    Devide series' area under N(0, 1) into `vocab_size` equal areas
    Returns a np.array, where symbol: cut 
    Use inverse umulative distribution function
    """
    probs = np.arange(0, vocab_size, 1) / vocab_size
    # cumulative prob. function
    return norm.ppf(probs)

# @deprecated
# use numpy instead (np.searchsorted(.))
def bin_search(val, arr):
    """
    Adapted binary search (left)
    if `val` is <= than `m` => compare with m-1, otherwise compare with m+1
    Find symbol representation
    Return index of symbol
    """
    l = 0
    r = arr.shape[0] - 1
    while l <= r:
        m = (l + r + 1) // 2
        if arr[m] <= val:
            # base case: m is right-most index
            if m + 1 == arr.shape[0]:
                return m
            # compare `val` with right neighbour
            elif val <= arr[m + 1]:
                return m
            l = m + 1 
        else:
            #base case: `val` is <= than 2nd value index
            if m <= 1:
                return 0
            # compare `val` with left neighbour
            elif val > arr[m - 1]:
                return  m - 1
            r = m - 1
    return m


def val2symbol(ts_paa, vocab_size):
    """
    Convert continuous time series values into discrete values, 
    using `vocab_size` discrete values
    """
    vocab = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], dtype=str)

    #vocab = vocab[:vocab_size]
    # compute breakpoints under a normal distribution ~ N(0, 1)
    breakpoints = get_breakpoints(vocab_size)
    # get ids for symbol conversion
    symbol_ids = np.searchsorted(breakpoints, ts_paa) - 1
    # convert ts to string
    ts_symbol = vocab[symbol_ids]

    return breakpoints, ts_symbol

def sax(ts, out_size, vocab_size, paa=True):
    """
    Apply SAX algorithm to time series, i.e. convert continuous values series into
    discrete values aggregated series
    :ts - time series of continuous values, numpy.array
    :out_size - the final output size of ts
    :vocab_size - number of sumbols to use (# lelvels), the size of vacabolary
    :paa - boolean variable, out_size is PAA if paa is True, out_size is Window size otherwise
    """
    if paa:
        paa_size = out_size
    else:
        paa_size = get_paa_size_from_window_size(ts.shape[0], out_size) 
    # Normalize series
    ts_norm = znorm(ts)
    # Convert normalized series to paa
    ts_paa = ts2paa(ts_norm, paa_size)
    # Convert paa series into symbols
    breakpoints, ts_sax = val2symbol(ts_paa, vocab_size)
    # Lookup table containing distance between symbols 
    dist_lookup_table = compute_dist_lookup_table(breakpoints)

    return breakpoints, dist_lookup_table, ts_norm, ts_paa, ts_sax

def symbol2index(ts_sax):
    """
    Converts symbol string to index values of symbols
    ts_sax: series as symbols, i.e. sax representation of a series
    """
    # lookup table for symbols' indeces
    s2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 
            'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 
            'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 
            'y': 24, 'z': 25, 'A': 26, 'B': 27, 'C': 28, 'D': 29, 'E': 30, 'F': 31, 
            'G': 32, 'H': 33, 'I': 34, 'J': 35, 'K': 36, 'L': 37, 'M': 38, 'N': 39, 
            'O': 40, 'P': 41, 'Q': 42, 'R': 43, 'S': 44, 'T': 45, 'U': 46, 'V': 47, 
            'W': 48, 'X': 49, 'Y': 50, 'Z': 51}
    # init. id series
    ts_id = np.empty(ts_sax.shape[0], dtype=int)
    # convert symbols to ids
    for i in range(ts_sax.shape[0]):
        ts_id[i] = s2id[ts_sax[i]]
    
    return ts_id

def get_dists(ts1_sax, ts2_sax, lookup_table):
    """
    Compute distance between each symbol of two words (series) using a lookup table
    ts1_sax and ts2_sax are two sax representations (strings) built under the same conditions
    """
    # Verify integrity
    if ts1_sax.shape[0] != ts2_sax.shape[0]:
        return -1
    # convert symbol series into series of indexes (symbol indexes)
    ts1_sax_id = symbol2index(ts1_sax)
    ts2_sax_id = symbol2index(ts2_sax)
    # array of distances between symbols
    dists = np.zeros(ts1_sax.shape[0])
    for i in range(ts1_sax_id.shape[0]):
        dists[i] = lookup_table[ts1_sax_id[i], ts2_sax_id[i]]
    
    return dists

def compute_mindist(n, lookup_table, ts1_sax, ts2_sax):
    """
    Minimum  distance  between  the  original  time  series  of  two  words
    `n` is the original series' length
    """
    aux = np.sqrt(n / ts1_sax.shape[0])
    dists = get_dists(ts1_sax, ts2_sax, lookup_table)
    dists_squares = np.square(dists)
    dists_sum_squares = np.sum(dists_squares)

    return aux * np.sqrt(dists_sum_squares)

def get_tightness_of_lower_bound(lookup_table, ts1, ts2, ts1_sax, ts2_sax):
    """
    Compute the tightness of the lower bound
    Used to find the parameters settings
    """
    # compute euclidean distance between original series
    or_dist = euclidean(ts1, ts2)
    # compute MINDIST for sax series
    mindist = compute_mindist(ts1.shape[0],lookup_table, ts1_sax, ts2_sax)

    return mindist / or_dist

def compute_dist_lookup_table(breakpoints):
    """
    The lookup table is computed as described in [X]
        d(r, c) = |0, if |r - c| <= 1
                  |abs(breakpoints[i] - breakpoints[j-1]), otherwise
    Contiguous values have distance 0, thus are not computed
    """
    # init. matrix
    lookup_table_dist = np.zeros((breakpoints.shape[0], breakpoints.shape[0]))
    # compute distances
    for bi in range(breakpoints.shape[0]):
        # increment by 2, since contiguous values have distance 0
        for bj in range(bi + 2, breakpoints.shape[0]):
            # since breakpoints[0] = - np.inf and symbol is conditioned by <=
            # bi is set to next value
            # compute distance
            dist = breakpoints[bj] - breakpoints[bi + 1]
            # set distance
            lookup_table_dist[bi, bj] = dist
            # mirror
            lookup_table_dist[bj, bi] = dist

    return lookup_table_dist

def get_paa_size_from_window_size(n, window_size):
    """
    Gets paa size from a sliding window size.
    Use sliding window instead of symbol series.
    """
    if n % window_size > 0:  
        return n // window_size + 1

    return n // window_size

###############################################################################################
###############################################################################################

def main(args):
    #CONSTATNS
    MIN_VOCAB_SIZE = 1
    MAX_VOCAB_SIZE = 52
    MIN_PAA_SIZE = 1
    ######################

    # Finding VOCAB_SIZE & PAA_SIZE. It is highly data dependent. Best values are those
    # which minimize the tightness of the lowers bound
    # Objective: Minimize(MINDIST(Â, Ê) / D(A, B)), i.e. Tightness of Lower Bound

    # Read data (skips header)
    data = np.loadtxt(args.data_path, delimiter=',', skiprows=1)
    df = pd.read_csv(args.data_path)
    data = df.as_matrix()
    cols = list(df.columns)
    #switch columns with rows (=>row is a time series)
    data = data.T
    #read arguments
    # n = len of series
    VOCAB_SIZE = args.vocab_size
    PAA_SIZE = args.paa_size
    WINDOW_SIZE =  args.window_size

    breakpoints_l = []
    lookup_table_l = []
    ts_norm_l = []
    ts_paa_l = []
    ts_sax_l = []

    st = time()

    print("Computing SAX...")
    for ts in data:
        # get number of obs.
        n = ts.shape
        #get PAA_SIZE or WINDOW_SIZE
        if WINDOW_SIZE > 0:
            PAA_SIZE = get_paa_size_from_window_size(n, WINDOW_SIZE)
        # compute sax
        breakpoints, lookup_table, ts_norm, ts_paa, ts_sax = sax(ts, PAA_SIZE, VOCAB_SIZE)
        #add to list
        breakpoints_l.append(breakpoints)
        lookup_table_l.append(lookup_table)
        ts_norm_l.append(ts_norm)
        ts_paa_l.append(ts_paa)
        ts_sax_l.append(ts_sax)

    n_series = data.shape[0]
    # compute TLS
    tbl_df = pd.DataFrame()
    edd_df = pd.DataFrame()
    lcs_df = pd.DataFrame()
    print("Computing TLS, Editing distance and LCS...")
    for i in range(n_series):
        tlb = np.zeros(n_series)
        edd = np.zeros(n_series)
        lcs_ = np.zeros(n_series)
        for j in range(i+1, n_series):
            tlb[j] = get_tightness_of_lower_bound(lookup_table, data[i], data[j], ts_sax_l[i],  ts_sax_l[j])
            edd[j] = edit_distance(ts_sax_l[i], ts_sax_l[j])
            lcs_[j] = lcs(ts_sax_l[i], ts_sax_l[j])
        tbl_df[cols[i]] = tlb
        edd_df[cols[i]] = edd
        lcs_df[cols[i]] = lcs_
    
    # compute distences 


    # store files
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    #TBL
    tbl_df.to_csv(os.path.join(args.out_path, "tlb.csv"), index=False)
    #edd
    edd_df.to_csv(os.path.join(args.out_path, "edd.csv"), index=False)
    #lcs
    lcs_df.to_csv(os.path.join(args.out_path, "lcs.csv"), index=False)
    #SAX
    sax_df = matrix_to_df(cols, ts_sax_l)
    sax_df.to_csv(os.path.join(args.out_path, "sax.csv"), index=False)
    #PAA
    paa_df = matrix_to_df(cols, ts_paa_l)
    paa_df.to_csv(os.path.join(args.out_path, "paa.csv"), index=False)
    
    print("Total time:", time() - st, "sec.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-dp', '--data_path', type=str, nargs='?', default="./sample100.csv")
    parser.add_argument('-paa', '--paa_size', type=int, nargs='?', default=9)
    parser.add_argument('-v', '--vocab_size', type=int, nargs='?', default=4)
    parser.add_argument('-w', '--window_size', type=int, nargs='?', default=0)
    parser.add_argument('-od', '--out_path', type=str, nargs='?', default="./results")
    args = parser.parse_args()

    main(args)
