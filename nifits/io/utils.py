import numpy as np

def get_uv(puparray):
    """
    Computes the uv baselines for a given pupil configuration.

    Args:
    uparray ``ArrayLike`` : [m] (n_tel, 2)
    """
    from itertools import combinations
    uv = []
    for pair in combinations(puparray, 2):
        #print(pair)
        uv.append(pair[0]-pair[1])
    uv = np.array(uv)
    indices = []
    for pair in combinations(np.arange(puparray.shape[0]), 2):
        #print(pair)
        indices.append((pair[0],pair[1]))
    indices = np.array(indices)
    return uv, indices
    
