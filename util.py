import numpy as np

def bilinear_interp_sampling(tensor, h, w, start_res=1024, end_res=4):
    h = (end_res - 1) * h / (start_res - 1)
    w = (end_res - 1) * w / (start_res - 1)

    h_ = np.floor(h).astype(np.int)
    w_ = np.floor(w).astype(np.int)

    h_ = np.minimum(h_, end_res - 2)
    w_ = np.minimum(w_, end_res - 2)

    dh = h - h_
    dw = w - w_

    z11 = tensor[:, : , h_, w_]
    z12 = tensor[:, : , h_ + 1 , w_]
    z21 = tensor[:, : , h_, w_ + 1]
    z22 = tensor[:, : , h_ + 1 , w_ + 1]

    a11 = (1 - dh) * (1 - dw)
    a21 = (dh) * (1 - dw)
    a12 = (1 - dh) * (dw)
    a22 = dh * dw

    v = z11 * a11 + z21 * a21 + z12 * a12 + z22 * a22

    return v
