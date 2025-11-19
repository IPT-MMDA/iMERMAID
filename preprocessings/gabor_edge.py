import numpy as np
import cv2

def gabor_edge(vv):
    vv_norm = cv2.normalize(vv, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # 1) Gabor bank (4 orientations)
    gabors = []
    for theta in (0, np.pi/4, np.pi/2, 3*np.pi/4):
        kern = cv2.getGaborKernel((21,21), sigma=5, theta=theta, lambd=10, gamma=0.5)
        gabors.append(cv2.filter2D(vv_norm, cv2.CV_32F, kern))
    min_resp = np.min(np.stack(gabors,axis=-1), axis=-1)
    # 2) Canny edges on intensity
    vv_u8 = (255*vv_norm).astype(np.uint8)
    edges = cv2.Canny(vv_u8, 30, 100) / 255.0
    # 3) Fill edges to thicker lines
    edges = cv2.dilate(edges.astype(np.float32), np.ones((3,3))) 
    # 4) Assemble channels
    ch0 = cv2.normalize(vv_norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ch1 = cv2.normalize(1 - min_resp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # high=flat
    ch2 = (255 * edges).astype(np.uint8)
    return cv2.merge([ch0, ch1, ch2])