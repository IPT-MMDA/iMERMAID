import numpy as np
import cv2

def db(src):
    vv = src
    vv[~np.isfinite(vv)] = -1
    orig = np.copy(vv)

    vv[vv <= 0] = np.max(vv)

    # vv = cv2.morphologyEx(vv, cv2.MORPH_CLOSE, np.ones((13, 13)))

    r = 20 * np.log10(vv)

    r = cv2.normalize(r, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    r[orig <= 0] = 255

    rgb = cv2.merge([r, r, r])

    return rgb