import numpy as np

def sample_xp(low,high,rng,Ns= 1,q_levels = 0):
    vals = np.linspace(low,high,q_levels)
    x1 = rng.integers(0,q_levels,1)
    x2 = rng.integers(0,q_levels,Ns - 1)
    x3 = np.concatenate((x1,x2))
    idx = np.mod(np.cumsum(x3),q_levels).astype(int)
    if Ns>1 and idx[0] == idx[-1]:
        all_indices = np.arange(q_levels)
        mask = (all_indices != idx[0] & (all_indices != idx[-2]))
        valid_indices = [mask]

        if len(valid_indices) > 0:
            ii = rng.integers(len(valid_indices))
            idx[-1] = valid_indices[ii]

    x = vals[idx]
    return x,idx