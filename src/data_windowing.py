import numpy as np

def create_bidirectional_windows(acc, gyr, quat, fs, window_size, stride):
    if window_size < 3 or stride < 1:
        raise ValueError("window_size >= 3 and stride >= 1 required.")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd for symmetric context.")
    N = acc.shape[0]
    half = window_size // 2
    Xg, Xa, F, Y = [], [], [], []
    for c in range(half, N - half, stride):
        i0, i1 = c - half, c + half + 1
        Xa.append(acc[i0:i1, :]); Xg.append(gyr[i0:i1, :])
        Y.append(quat[c, :]); F.append([fs])
    return (np.asarray(Xg, np.float32),
            np.asarray(Xa, np.float32),
            np.asarray(F,  np.float32),
            np.asarray(Y,  np.float32))
