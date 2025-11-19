import numpy as np
import cv2

def lee_filter(img, win=7):
    """Adaptive Lee speckle filter."""
    img_mean = cv2.blur(img, (win, win))
    img_sqr_mean = cv2.blur(img*img, (win, win))
    img_var = img_sqr_mean - img_mean**2
    overall_var = np.var(img)
    weight = img_var / (img_var + overall_var + 1e-6)
    return img_mean + weight*(img - img_mean)

def lee_variance(vv):
    # 1) Speckle filter
    vv_filt = lee_filter(vv.astype(np.float32), win=7)
    # 2) Local variance map (block size = 32)
    k = 32
    mean = cv2.blur(vv_filt, (k, k))
    sq_mean = cv2.blur(vv_filt**2, (k, k))
    var_map = sq_mean - mean**2
    # 3) Normalize channels to 0–255
    i_ch  = cv2.normalize(vv_filt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    v_ch  = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # 4) Synthetic third channel = log‑ratio of local to global mean
    global_mean = np.mean(vv_filt)
    r_ch  = np.clip(50 * np.log1p(vv_filt / (global_mean+1e-6)), 0, 255).astype(np.uint8)
    return cv2.merge([i_ch, v_ch, r_ch])