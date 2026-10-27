"""Counter-timing QC rule applied to the hospital-level series (Section S1, rule v).

A pair of adjacent intervals is rejected when one lies below 0.75 and the other above 1.25 times the mean of
the two outer neighbouring intervals, while the pair sum is within 15% of twice that mean: energy has been
logged one interval late, so the pair total is plausible but neither interval is."""
import numpy as np, pandas as pd

LO, HI, TOL = 0.75, 1.25, 0.15


def timing_pairs(L):
    """Boolean mask of the intervals (both members of each pair) flagged in the load series L."""
    a, b = L, L.shift(-1)
    nb = (L.shift(1) + L.shift(-2))/2
    first = (((a < LO*nb) & (b > HI*nb)) | ((a > HI*nb) & (b < LO*nb))) & ((a + b - 2*nb).abs() < TOL*2*nb) & (nb > 0)
    return first | first.shift(1, fill_value=False)


def reject_timing_pairs(H, cols=('E', 'L', 'B', 'X')):
    """Set the flagged hospital intervals to NaN; returns the number of intervals rejected."""
    bad = timing_pairs(H['L']).values
    H.loc[bad, list(cols)] = np.nan
    pd.Series(H.index[bad]).to_csv('tab_qc_timing.csv', index=False, header=['interval_end_local_time'])
    return int(bad.sum())
