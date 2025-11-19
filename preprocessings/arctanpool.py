import numpy as np
import cv2

def cv2_pooling(src):
    stride = 32
    mean_map = cv2.filter2D(src, -1, np.ones((stride, stride)) / stride**2)
    std_map = cv2.filter2D(src**2 - 2 * src * mean_map + mean_map**2, -1, np.ones((stride, stride)) / stride**2)

    pooled_map = mean_map[::stride, ::stride]
    pooled_map_std = np.maximum(5e-3, std_map[::stride, ::stride])

    dst_shape = (src.shape[1], src.shape[0])

    pooled_map = cv2.GaussianBlur(pooled_map, (7, 7), 0)
    pooled_map = cv2.resize(pooled_map, dst_shape)

    pooled_map_std = cv2.GaussianBlur(pooled_map_std, (7, 7), 0)
    pooled_map_std = cv2.resize(pooled_map_std, dst_shape)

    return (src - pooled_map) / pooled_map_std

def arctanpool(src):
    vv = src
    vv[~np.isfinite(vv)] = -1
    orig = np.copy(vv)

    valid = vv[vv > 0]
    mean = np.mean(valid)
    std = np.std(valid)

    vv[vv <= 0] = np.max(vv)

    # vv = cv2.morphologyEx(vv, cv2.MORPH_CLOSE, np.ones((13, 13)))

    r = 20 * np.log10(vv)
    g = (np.arctan((vv - mean) / (std)) + 0.5 * np.pi) / np.pi
    b = (np.arctan(cv2_pooling(vv) / 2) + 0.5 * np.pi) / np.pi

    r = cv2.normalize(r, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    g = (255 * g).astype("uint8")
    b = (255 * b).astype("uint8")

    r[orig <= 0] = 255
    g[orig <= 0] = 255
    b[orig <= 0] = 255

    rgb = cv2.merge([r, g, b])

    return rgb